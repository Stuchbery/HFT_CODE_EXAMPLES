#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "INDICATOR_MODULES.hpp"

int main()
{
    // Example usage
    MultiWeightedLinearRegression regression(2,0.9999);

    std::vector<double> explanatoryValues = {1.0, 2.0};
    double targetValue = 3.0;

    regression.update_WLR(explanatoryValues, targetValue);

    // You can continue to use the regression object for further updates or calculations

    return 0;
}
