## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 79.4085105772
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612)
1: (-59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992)
2: (-53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439)
3: (-58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935)
4: (-66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492)
5: (-60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872)
6: (-95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498)
7: (-70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839)
8: (-77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492)
9: (-64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648)
10: (-95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122)
11: (-91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326)
12: (-91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001)
13: (-97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521)
14: (-141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317)
15: (-75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994)
16: (-96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474)
17: (-139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440)
18: (-90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867)
19: (-72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048)
20: (-67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908)
21: (-89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444)
22: (-90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815)
23: (-71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896)
24: (-86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379)
25: (-76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524)
26: (-101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566)
27: (-88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820)
28: (-71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006)
29: (-92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924)
30: (-88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199)
31: (-93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035)
32: (-93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315)
33: (-121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873)
34: (-100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062)
35: (-98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817)
36: (-100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760)
37: (-141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120)
38: (-119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850)
39: (-135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140)
40: (-112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437)
41: (-94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704)
42: (-69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776)

## BASE Result
execution time: IAR + LP analysis = 2.73 + 96.99 = 99.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -83.2047295, upper bound: 83.2047295


# Binary Search by BASE starts (time budget: 17900.28 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=124.9945068359375
rel_dist={1: [-79.47797671394284, 79.47797670615168]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=124.9945068359375
rel_dist={1: [-76.66888544603012, 76.66888544397699]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=124.9945068359375
rel_dist={1: [-77.72391643090478, 77.72391642674157]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=124.9945068359375
rel_dist={1: [-78.64399325943418, 78.64399325111157]}

## Binary Search Result
Binary search time: 512.49 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_random_Z) starts
Time budget: 17387.78 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 646

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1385

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5528992, upper bound: 81.5544070
time: 117.20 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5544070, upper bound: 81.5528992
time: 141.08 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 258.30 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 258.30
Output dim: 1, lower bound: -81.5528992, upper bound: 81.5544070
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 258.30
Output dim: 1, lower bound: -81.5544070, upper bound: 81.5528992

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 678

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5424299, upper bound: 81.5450926
time: 94.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5435893, upper bound: 81.5439340
time: 101.60 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1369

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 639

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5200003, upper bound: 81.5457212
time: 100.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5472318, upper bound: 81.5184761
time: 144.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 247.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 247.70
Output dim: 1, lower bound: -81.5424299, upper bound: 81.5450926
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 247.70
Output dim: 1, lower bound: -81.5435893, upper bound: 81.5439340
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 247.70
Output dim: 1, lower bound: -81.5200003, upper bound: 81.5457212
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 247.70
Output dim: 1, lower bound: -81.5472318, upper bound: 81.5184761

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 865

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1564

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5423289, upper bound: 81.5450150
time: 101.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5423528, upper bound: 81.5449901
time: 103.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 522

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5432370, upper bound: 81.5370902
time: 160.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5367309, upper bound: 81.5435835
time: 99.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1694

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1608

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5149955, upper bound: 81.5447764
time: 105.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5190550, upper bound: 81.5407019
time: 119.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1557

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1571

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5462302, upper bound: 81.4966999
time: 134.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5254740, upper bound: 81.5174873
time: 96.05 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 232.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 232.46
Output dim: 1, lower bound: -81.5423289, upper bound: 81.5450150
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 232.46
Output dim: 1, lower bound: -81.5423528, upper bound: 81.5449901
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 232.46
Output dim: 1, lower bound: -81.5432370, upper bound: 81.5370902
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 232.46
Output dim: 1, lower bound: -81.5367309, upper bound: 81.5435835
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 232.46
Output dim: 1, lower bound: -81.5149955, upper bound: 81.5447764
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 232.46
Output dim: 1, lower bound: -81.5190550, upper bound: 81.5407019
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 232.46
Output dim: 1, lower bound: -81.5462302, upper bound: 81.4966999
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 232.46
Output dim: 1, lower bound: -81.5254740, upper bound: 81.5174873

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1573

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1684

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5415126, upper bound: 81.5415361
time: 123.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5387770, upper bound: 81.5442151
time: 114.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 797

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5286040, upper bound: 81.5440268
time: 100.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5414022, upper bound: 81.5313151
time: 111.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 760

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 941

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5325994, upper bound: 81.5366477
time: 145.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5427964, upper bound: 81.5264067
time: 159.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1429

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1622

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4906803, upper bound: 81.4976225
time: 165.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4906803, upper bound: 81.4976225
time: 108.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1603

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 740

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5124911, upper bound: 81.5230701
time: 113.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4933056, upper bound: 81.5422681
time: 637.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1669

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1735

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5129263, upper bound: 81.4982309
time: 100.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.4765887, upper bound: 81.5345625
time: 147.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 576

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5294238, upper bound: 81.4788063
time: 141.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -81.5284228, upper bound: 81.4798146
time: 140.21 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 284.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.5415126, upper bound: 81.5415361
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.5387770, upper bound: 81.5442151
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.5286040, upper bound: 81.5440268
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.5414022, upper bound: 81.5313151
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.5325994, upper bound: 81.5366477
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.5427964, upper bound: 81.5264067
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.4906803, upper bound: 81.4976225
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.4906803, upper bound: 81.4976225
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.5124911, upper bound: 81.5230701
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.4933056, upper bound: 81.5422681
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.5129263, upper bound: 81.4982309
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.4765887, upper bound: 81.5345625
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.5294238, upper bound: 81.4788063
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 284.41
Output dim: 1, lower bound: -81.5284228, upper bound: 81.4798146
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 284.41
Output dim: 1, lower bound: -81.5254740, upper bound: 81.5174873
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=124.9945068359375
rel_dist={1: [-81.5568719651244, 81.5568719651244]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1541

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 526

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2409552, upper bound: 80.2399111
time: 116.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2399111, upper bound: 80.2409552
time: 98.91 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 215.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 215.35
Output dim: 1, lower bound: -80.2409552, upper bound: 80.2399111
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 215.35
Output dim: 1, lower bound: -80.2399111, upper bound: 80.2409552

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 861

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2273142, upper bound: 80.2389793
time: 128.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2399858, upper bound: 80.2273142
time: 135.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1543

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2362810, upper bound: 80.2405324
time: 109.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2362810, upper bound: 80.2373433
time: 112.09 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 224.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 224.13
Output dim: 1, lower bound: -80.2273142, upper bound: 80.2389793
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 224.13
Output dim: 1, lower bound: -80.2399858, upper bound: 80.2273142
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 224.13
Output dim: 1, lower bound: -80.2362810, upper bound: 80.2405324
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 224.13
Output dim: 1, lower bound: -80.2362810, upper bound: 80.2373433

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1573

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2246339, upper bound: 80.2165177
time: 132.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2059958, upper bound: 80.2352158
time: 113.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 750

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 760

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2152185, upper bound: 80.2054063
time: 90.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2179938, upper bound: 80.2026118
time: 167.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2362550, upper bound: 80.2277689
time: 127.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2234973, upper bound: 80.2405066
time: 100.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1696

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2386243, upper bound: 80.2335945
time: 133.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2325292, upper bound: 80.2364830
time: 92.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 229.01 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 229.01
Output dim: 1, lower bound: -80.2246339, upper bound: 80.2165177
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 229.01
Output dim: 1, lower bound: -80.2059958, upper bound: 80.2352158
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 229.01
Output dim: 1, lower bound: -80.2152185, upper bound: 80.2054063
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 229.01
Output dim: 1, lower bound: -80.2179938, upper bound: 80.2026118
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 229.01
Output dim: 1, lower bound: -80.2362550, upper bound: 80.2277689
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 229.01
Output dim: 1, lower bound: -80.2234973, upper bound: 80.2405066
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 229.01
Output dim: 1, lower bound: -80.2386243, upper bound: 80.2335945
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 229.01
Output dim: 1, lower bound: -80.2325292, upper bound: 80.2364830

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 717

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2243281, upper bound: 80.2157420
time: 109.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2238497, upper bound: 80.2162072
time: 113.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 823

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2041408, upper bound: 80.2307219
time: 135.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2014979, upper bound: 80.2333619
time: 136.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1617

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2145489, upper bound: 80.1952427
time: 115.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2050991, upper bound: 80.2047359
time: 460.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 559

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1466335, upper bound: 80.1251813
time: 238.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1411467, upper bound: 80.1306948
time: 101.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1702

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1663

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2332312, upper bound: 80.1948143
time: 420.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.2032689, upper bound: 80.2247511
time: 128.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1690

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1791826, upper bound: 80.1964658
time: 101.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -80.1791826, upper bound: 80.1964658
time: 101.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 205.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 205.98
Output dim: 1, lower bound: -80.2243281, upper bound: 80.2157420
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 205.98
Output dim: 1, lower bound: -80.2238497, upper bound: 80.2162072
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 205.98
Output dim: 1, lower bound: -80.2041408, upper bound: 80.2307219
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 205.98
Output dim: 1, lower bound: -80.2014979, upper bound: 80.2333619
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 205.98
Output dim: 1, lower bound: -80.2145489, upper bound: 80.1952427
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 205.98
Output dim: 1, lower bound: -80.2050991, upper bound: 80.2047359
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 205.98
Output dim: 1, lower bound: -80.1466335, upper bound: 80.1251813
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 205.98
Output dim: 1, lower bound: -80.1411467, upper bound: 80.1306948
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 205.98
Output dim: 1, lower bound: -80.2332312, upper bound: 80.1948143
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 205.98
Output dim: 1, lower bound: -80.2032689, upper bound: 80.2247511
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 205.98
Output dim: 1, lower bound: -80.1791826, upper bound: 80.1964658
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 205.98
Output dim: 1, lower bound: -80.1791826, upper bound: 80.1964658
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.98
Output dim: 1, lower bound: -80.2386243, upper bound: 80.2335945
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 205.98
Output dim: 1, lower bound: -80.2325292, upper bound: 80.2364830
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=124.9945068359375
rel_dist={1: [-80.2411023357962, 80.2411023398264]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 532

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 899

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4572458, upper bound: 79.4742061
time: 125.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4742061, upper bound: 79.4572458
time: 104.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 229.43 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 229.43
Output dim: 1, lower bound: -79.4572458, upper bound: 79.4742061
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 229.43
Output dim: 1, lower bound: -79.4742061, upper bound: 79.4572458

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 963

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4570376, upper bound: 79.4523819
time: 120.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4354127, upper bound: 79.4739979
time: 116.82 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1549

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1368

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4669774, upper bound: 79.4531823
time: 106.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4701396, upper bound: 79.4500200
time: 139.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 248.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 248.11
Output dim: 1, lower bound: -79.4570376, upper bound: 79.4523819
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 248.11
Output dim: 1, lower bound: -79.4354127, upper bound: 79.4739979
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 248.11
Output dim: 1, lower bound: -79.4669774, upper bound: 79.4531823
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 248.11
Output dim: 1, lower bound: -79.4701396, upper bound: 79.4500200

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 940

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 752

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4531002, upper bound: 79.4474802
time: 99.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4521281, upper bound: 79.4484473
time: 132.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1368

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 984

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.3890341, upper bound: 79.4620489
time: 158.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4234582, upper bound: 79.4276132
time: 84.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 703

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 837

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4537928, upper bound: 79.4526862
time: 160.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4664818, upper bound: 79.4399930
time: 96.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 941

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4688766, upper bound: 79.4345561
time: 127.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4546830, upper bound: 79.4487570
time: 98.08 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 227.89 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 227.89
Output dim: 1, lower bound: -79.4531002, upper bound: 79.4474802
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 227.89
Output dim: 1, lower bound: -79.4521281, upper bound: 79.4484473
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 227.89
Output dim: 1, lower bound: -79.3890341, upper bound: 79.4620489
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 227.89
Output dim: 1, lower bound: -79.4234582, upper bound: 79.4276132
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 227.89
Output dim: 1, lower bound: -79.4537928, upper bound: 79.4526862
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 227.89
Output dim: 1, lower bound: -79.4664818, upper bound: 79.4399930
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 227.89
Output dim: 1, lower bound: -79.4688766, upper bound: 79.4345561
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 227.89
Output dim: 1, lower bound: -79.4546830, upper bound: 79.4487570

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 688

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1542

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4526117, upper bound: 79.4451200
time: 132.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4507533, upper bound: 79.4469913
time: 123.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 851

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4470717, upper bound: 79.4480222
time: 92.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4517020, upper bound: 79.4433694
time: 91.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1480

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 840

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.3868474, upper bound: 79.4597118
time: 132.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.3867014, upper bound: 79.4598586
time: 115.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1548

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1679

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4220300, upper bound: 79.3961053
time: 90.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.3575291, upper bound: 79.4261814
time: 118.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1562

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4373118, upper bound: 79.4517005
time: 139.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4528078, upper bound: 79.4362070
time: 129.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 543

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 638

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4226550, upper bound: 79.4098320
time: 106.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4236342, upper bound: 79.3961553
time: 111.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1385

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 892

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4373893, upper bound: 79.4330649
time: 95.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4673915, upper bound: 79.4030596
time: 118.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 837

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 625

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4308281, upper bound: 79.4424178
time: 123.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4485764, upper bound: 79.4418757
time: 99.81 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 225.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4526117, upper bound: 79.4451200
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4507533, upper bound: 79.4469913
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4470717, upper bound: 79.4480222
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4517020, upper bound: 79.4433694
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.3868474, upper bound: 79.4597118
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.3867014, upper bound: 79.4598586
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4220300, upper bound: 79.3961053
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.3575291, upper bound: 79.4261814
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4373118, upper bound: 79.4517005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4528078, upper bound: 79.4362070
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4226550, upper bound: 79.4098320
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4236342, upper bound: 79.3961553
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4373893, upper bound: 79.4330649
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4673915, upper bound: 79.4030596
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4308281, upper bound: 79.4424178
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 225.83
Output dim: 1, lower bound: -79.4485764, upper bound: 79.4418757

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 964

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4517717, upper bound: 79.4399755
time: 151.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4442299, upper bound: 79.4439080
time: 102.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.4922180, 78.6617432, -106.4922180, 78.6617432, -185.1539612, 185.1539612
1: -59.2393761, 65.7551270, -59.2393761, 65.7551270, -124.9945068, 124.9944992
2: -53.0345535, 63.3313904, -53.0345535, 63.3313904, -116.3659439, 116.3659439
3: -58.3790474, 72.7411499, -58.3790474, 72.7411499, -131.1201935, 131.1201935
4: -66.7221832, 73.9994659, -66.7221832, 73.9994659, -140.7216492, 140.7216492
5: -60.4717331, 70.5876770, -60.4717331, 70.5876770, -131.0594025, 131.0593872
6: -95.4778061, 62.5982475, -95.4778061, 62.5982475, -158.0760498, 158.0760498
7: -70.7483521, 67.5726318, -70.7483521, 67.5726318, -138.3209839, 138.3209839
8: -77.0471497, 90.0951996, -77.0471497, 90.0951996, -167.1423492, 167.1423492
9: -64.8811722, 71.3190002, -64.8811722, 71.3190002, -136.2001648, 136.2001648
10: -95.5132599, 90.8742599, -95.5132599, 90.8742599, -186.3874969, 186.3875122
11: -91.8675995, 56.0995331, -91.8675995, 56.0995331, -147.9671326, 147.9671326
12: -91.0567017, 72.7958984, -91.0567017, 72.7958984, -163.8526001, 163.8526001
13: -97.1215668, 95.7406006, -97.1215668, 95.7406006, -192.8621674, 192.8621521
14: -141.9139404, 80.1978912, -141.9139404, 80.1978912, -222.1118164, 222.1118317
15: -75.4594116, 67.2486801, -75.4594116, 67.2486801, -142.7080994, 142.7080994
16: -96.1752625, 69.5689850, -96.1752625, 69.5689850, -165.7442474, 165.7442474
17: -139.3930054, 76.3171539, -139.3930054, 76.3171539, -215.7101593, 215.7101440
18: -90.9892807, 75.4321136, -90.9892807, 75.4321136, -166.4213715, 166.4213867
19: -72.7950134, 45.6102028, -72.7950134, 45.6102028, -118.4052124, 118.4052048
20: -67.6466827, 52.5951080, -67.6466827, 52.5951080, -120.2417831, 120.2417908
21: -89.1179657, 53.5706825, -89.1179657, 53.5706825, -142.6886444, 142.6886444
22: -90.9661942, 55.0910873, -90.9661942, 55.0910873, -146.0572815, 146.0572815
23: -71.4414062, 55.7566833, -71.4414062, 55.7566833, -127.1980896, 127.1980896
24: -86.1005478, 51.3332825, -86.1005478, 51.3332825, -137.4338379, 137.4338379
25: -76.2652740, 58.9671860, -76.2652740, 58.9671860, -135.2324524, 135.2324524
26: -101.2129440, 82.2951050, -101.2129440, 82.2951050, -183.5080566, 183.5080566
27: -88.7444839, 59.8771019, -88.7444839, 59.8771019, -148.6215820, 148.6215820
28: -71.7643585, 60.9762421, -71.7643585, 60.9762421, -132.7406006, 132.7406006
29: -92.3658218, 48.6994705, -92.3658218, 48.6994705, -141.0652924, 141.0652924
30: -88.5172882, 65.4868317, -88.5172882, 65.4868317, -154.0041199, 154.0041199
31: -93.3612213, 55.1560822, -93.3612213, 55.1560822, -148.5173035, 148.5173035
32: -93.1964798, 59.2045593, -93.1964798, 59.2045593, -152.4010315, 152.4010315
33: -121.6869659, 78.7931213, -121.6869659, 78.7931213, -200.4800873, 200.4800873
34: -100.7623367, 57.9915771, -100.7623367, 57.9915771, -158.7539062, 158.7539062
35: -98.9166794, 60.6471062, -98.9166794, 60.6471062, -159.5637817, 159.5637817
36: -100.4784012, 62.8444786, -100.4784012, 62.8444786, -163.3228760, 163.3228760
37: -141.7263336, 62.0128784, -141.7263336, 62.0128784, -203.7392120, 203.7392120
38: -119.7381058, 78.6489792, -119.7381058, 78.6489792, -198.3870850, 198.3870850
39: -135.0441895, 75.5595322, -135.0441895, 75.5595322, -210.6036987, 210.6037140
40: -112.0002518, 61.2994995, -112.0002518, 61.2994995, -173.2997437, 173.2997437
41: -94.1975403, 64.6577301, -94.1975403, 64.6577301, -158.8552704, 158.8552704
42: -69.2248230, 60.0297546, -69.2248230, 60.0297546, -129.2545776, 129.2545776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=677, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4456336, upper bound: 79.4395395
time: 98.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -79.4432970, upper bound: 79.4418736
time: 91.74 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 192.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 192.53
Output dim: 1, lower bound: -79.4517717, upper bound: 79.4399755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 192.53
Output dim: 1, lower bound: -79.4442299, upper bound: 79.4439080
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 192.53
Output dim: 1, lower bound: -79.4456336, upper bound: 79.4395395
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 192.53
Output dim: 1, lower bound: -79.4432970, upper bound: 79.4418736
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.4470717, upper bound: 79.4480222
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.4517020, upper bound: 79.4433694
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.3868474, upper bound: 79.4597118
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.3867014, upper bound: 79.4598586
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.4220300, upper bound: 79.3961053
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.3575291, upper bound: 79.4261814
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.4373118, upper bound: 79.4517005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.4528078, upper bound: 79.4362070
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.4226550, upper bound: 79.4098320
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.4236342, upper bound: 79.3961553
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.4373893, upper bound: 79.4330649
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.4673915, upper bound: 79.4030596
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.4308281, upper bound: 79.4424178
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 192.53
Output dim: 1, lower bound: -79.4485764, upper bound: 79.4418757
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=124.9945068359375
rel_dist={1: [-79.47797671394284, 79.47797670615168]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 12284.66 seconds
