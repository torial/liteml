from math import sqrt
# L1 (Manhattan) distance
def l1_distance(loc1, loc2):
    distance = sum((abs(loc1i - loc2i)
                    for loc1i, loc2i in zip(loc1, loc2)))
    return distance

# L2 (Eculidean) distance
def l2_distance(loc1, loc2):
    distance = sqrt(sum((loc1i - loc2i)**2
                    for loc1i, loc2i in zip(loc1, loc2)))
    return distance