import math
import random

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('dark_background')
matplotlib.use('Qt5Agg')
# Read data from file

class MultiWeightedLinearRegression:
    def __init__(self, num_explanatory_values, alpha=0.9999):
        self.num_rows = num_explanatory_values + 1
        self.run_once_WLR = 0
        self.alpha_WLR = alpha
        self.ESTIMATED_VALUE = 0
        self.PREV_TARGET_VALUE = 1
        self.PREV_EXPLANATORY_VALUES = []
        self.curr_timestamp = 0
        self.INVERSE_M = []
        self.ESTIMATE_B = []
        self.M = []
        self.V = []

    def update_WLR(self, EXPLANATORY_VALUES, TARGET_VALUE):

        for i in EXPLANATORY_VALUES:
            if math.isnan(i):
                return
        if math.isnan(TARGET_VALUE):
            return
        if self.run_once_WLR == 1:
            if (
                not np.array_equal(self.PREV_EXPLANATORY_VALUES, EXPLANATORY_VALUES)
                or self.PREV_TARGET_VALUE != TARGET_VALUE
            ):
                # Update M matrix
                for i in range(self.num_rows):
                    for j in range(self.num_rows):
                        if i == 0 and j == 0:
                            self.M[i][j] = 1 * (1 - self.alpha_WLR) + self.M[i][j] * self.alpha_WLR
                        elif i == 0:
                            self.M[i][j] = (
                                EXPLANATORY_VALUES[j - 1] * (1 - self.alpha_WLR)
                                + self.M[i][j] * self.alpha_WLR
                            )
                        elif j == 0:
                            self.M[i][j] = (
                                EXPLANATORY_VALUES[i - 1] * (1 - self.alpha_WLR)
                                + self.M[i][j] * self.alpha_WLR
                            )
                        else:
                            self.M[i][j] = (
                                EXPLANATORY_VALUES[i - 1]
                                * EXPLANATORY_VALUES[j - 1]
                                * (1 - self.alpha_WLR)
                                + self.M[i][j] * self.alpha_WLR
                            )

                # Update V matrix
                for i in range(self.num_rows):
                    if i == 0:
                        self.V[i][0] = (
                            self.alpha_WLR * self.V[i][0]
                            + (1 - self.alpha_WLR) * (1 * TARGET_VALUE)
                        )
                    else:
                        self.V[i][0] = (
                            self.alpha_WLR * self.V[i][0]
                            + (1 - self.alpha_WLR) * (EXPLANATORY_VALUES[i - 1] * TARGET_VALUE)
                        )

                # Update inverse of M matrix
                #self.INVERSE_M = np.linalg.inv(np.array(self.M))
                self.INVERSE_M = np.linalg.pinv(np.array(self.M))

                # Update ESTIMATE_B matrix
                for i in range(self.num_rows):
                    sum = 0
                    for j in range(self.num_rows):
                        sum += self.INVERSE_M[i][j] * self.V[j][0]
                    self.ESTIMATE_B[i][0] = sum

                # Calculate estimated value
                self.ESTIMATED_VALUE = 0
                for i in range(self.num_rows):
                    if i == 0:
                        self.ESTIMATED_VALUE += self.ESTIMATE_B[i][0] * 1
                    else:
                        self.ESTIMATED_VALUE += EXPLANATORY_VALUES[i - 1] * self.ESTIMATE_B[i][0]

                self.PREV_EXPLANATORY_VALUES = EXPLANATORY_VALUES
                self.PREV_TARGET_VALUE = TARGET_VALUE
        else:
            self.M = [[0] * self.num_rows for _ in range(self.num_rows)]
            self.V = [[0] for _ in range(self.num_rows)]
            self.INVERSE_M = [[0] * self.num_rows for _ in range(self.num_rows)]
            self.ESTIMATE_B = [[0] for _ in range(self.num_rows)]

            # Initialize M matrix
            for i in range(self.num_rows):
                for j in range(self.num_rows):
                    if i == 0 and j == 0:
                        self.M[i][j] = 1
                    elif i == 0:
                        self.M[i][j] = EXPLANATORY_VALUES[j - 1]
                    elif j == 0:
                        self.M[i][j] = EXPLANATORY_VALUES[i - 1]
                    else:
                        self.M[i][j] = EXPLANATORY_VALUES[i - 1] * EXPLANATORY_VALUES[j - 1]

            # Initialize V matrix
            for i in range(self.num_rows):
                if i == 0:
                    self.V[i][0] = 1 * TARGET_VALUE
                else:
                    self.V[i][0] = EXPLANATORY_VALUES[i - 1] * TARGET_VALUE

            self.run_once_WLR = 1

    def calc_WLR(self, EXPLANATORY_VALUES):
        if self.run_once_WLR == 1:
            # Calculate estimated value
            self.ESTIMATED_VALUE = 0
            for i in range(self.num_rows):
                if i == 0:
                    self.ESTIMATED_VALUE += self.ESTIMATE_B[i][0] * 1
                else:
                    self.ESTIMATED_VALUE += EXPLANATORY_VALUES[i - 1] * self.ESTIMATE_B[i][0]

    def setAlphaByNbOfUpdates(self, nbUpdates):
        self.alpha_W


example_data = [] # Test data with known coefficients

#Create Test data
true_coefficients = np.array([1.5, 2.0, 3.0]) # True coefficients
for _ in range(10000):
    x_values = np.array([random.uniform(1, 10) for _ in range(3)]) # Generate random x values
    y_value = np.dot(x_values, true_coefficients) + random.uniform(-2, 2) # Calculate y value using true coefficients
    example_data.append((x_values, y_value))

WLR_obj = MultiWeightedLinearRegression(3, alpha=0.99999) # Initialize MultiWeightedLinearRegression object

#Push Test data though Multidiamensional WeightedLinearRegression with expotential forgetting
for data_point in example_data:
    x_values, y_value = data_point
    WLR_obj.update_WLR(np.asarray(x_values), y_value)  # Feed the test data into the update_WLR function

est_coefficients = WLR_obj.ESTIMATE_B # Retrieve estimated coefficients

print("True Coefficients:")
for i in range(len(true_coefficients)):
    print(f"True B{i}: {true_coefficients[i]}")

print("\nEstimated Coefficients:")
print(f"Estimated Linear Intercept: {est_coefficients[0][0]}")

for i in range(1,len(est_coefficients)):
    print(f"Estimated B{i - 1}: {est_coefficients[i][0]}")


#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#ax1.scatter(ts_lst, high_price_TPM, c='orange')
#ax2.plot(ts_lst, ALPHA_SIG_lst, c='red'))
#plt.show()
