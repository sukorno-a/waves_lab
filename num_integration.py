"""Script to perform numerical integration of a semicircle by using the 'rectangle rule'"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
sn.set_style("darkgrid")

def rect(x, y):
    """Numerical integration by plotting a series of thin rectangles and summing their area under a curve
    
    Parameters
    ----------
    x : array
        X co-ordinates
    y : array
        Y co-ordinates
        
    Returns
    -------
    area : float
        Total estimated area under the curve based on the sum of the rectangles"""

    x_range = max(x) - min(x)
    width = x_range / (len(x) - 1)
    area = 0

    plt.subplot()
    plt.plot(x, y, ms=0.1)
    # plt.scatter(x, y, linewidths=0.1)
    plt.xlabel("x")
    plt.ylabel("y")

    for i in range(len(x) - 1):
        mean_height = (y[i] + y[i + 1]) / 2
        rect_area = mean_height * width
        area += rect_area
        plt.bar(x[i], mean_height, width=width, align="edge", color = "lightblue", edgecolor = "slategray")

    plt.show()

    return area


# plot and integrate low res semicircle using provided txt file
# x1, y1 = np.loadtxt("task_1.3_semicircle_low.txt", unpack=True, skiprows=1)

# plt.figure()
# plt.plot(x1, y1, ms=1)
# plt.scatter(x1, y1, linewidths=0.1)
# plt.show()
# print(f"Estimated area: {rect(x1, y1)}")

# plot and integrate high res semicircle using provided txt file
# x2, y2 = np.loadtxt("task_1.3_semicircle_high.txt", unpack=True, skiprows=1)
# plt.figure()
# plt.plot(x2, y2, ms=1)
# plt.scatter(x2, y2, linewidths=0.1)
# plt.show()
# print(f"Estimated area: {rect(x2, y2)}")

# radius = 1
# circle_area = np.pi * (radius**2)
# print(f"Actual area: {circle_area/2}")

# final results show that area obtained by low-res data is 1.552, while area from high-res data is 1.57
# our analytical answer is of course pi * r^2 / 2 which is 1.571