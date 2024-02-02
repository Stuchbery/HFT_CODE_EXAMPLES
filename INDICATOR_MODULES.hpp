
double getDeterminant(const std::vector<std::vector<double>>& matrix){
        //cout << "getDeterminant" <<endl;
        std::size_t n = matrix.size();

        if (n != matrix[0].size())
        {
            throw std::runtime_error("Matrix is not quadratic");
        }

        // Create a copy of the input matrix
        std::vector<std::vector<double>> luMatrix(matrix);

        // Perform LU decomposition
        for (std::size_t k = 0; k < n - 1; k++)
        {
            for (std::size_t i = k + 1; i < n; i++)
            {
                if (luMatrix[k][k] == 0)
                {
                    throw std::runtime_error("Matrix is singular");
                }

                luMatrix[i][k] /= luMatrix[k][k];

                for (std::size_t j = k + 1; j < n; j++)
                {
                    luMatrix[i][j] -= luMatrix[i][k] * luMatrix[k][j];
                }
            }
        }

        // Calculate the determinant
        double determinant = 1.0;
        for (std::size_t i = 0; i < n; i++)
        {
            determinant *= luMatrix[i][i];
        }


        //cout << "getDeterminant Done" <<endl;
        return determinant;
    }
std::vector<std::vector<double>> getTranspose(const std::vector<std::vector<double>> matrix1) {

    //Transpose-matrix: height = width(matrix), width = height(matrix)
    std::vector<std::vector<double>> solution(matrix1[0].size(), std::vector<double> (matrix1.size()));

    //Filling solution-matrix
    for(size_t i = 0; i < matrix1.size(); i++) {
        for(size_t j = 0; j < matrix1[0].size(); j++) {
            solution[j][i] = matrix1[i][j];
        }
    }
    return solution;
}
std::vector<std::vector<double>> getCofactor(const std::vector<std::vector<double>> vect) {
    if(vect.size() != vect[0].size()) {
        throw std::runtime_error("Matrix is not quadratic");
    }

    std::vector<std::vector<double>> solution(vect.size(), std::vector<double> (vect.size()));
    std::vector<std::vector<double>> subVect(vect.size() - 1, std::vector<double> (vect.size() - 1));

    for(std::size_t i = 0; i < vect.size(); i++) {
        for(std::size_t j = 0; j < vect[0].size(); j++) {

            int p = 0;
            for(size_t x = 0; x < vect.size(); x++) {
                if(x == i) {
                    continue;
                }
                int q = 0;

                for(size_t y = 0; y < vect.size(); y++) {
                    if(y == j) {
                        continue;
                    }

                    subVect[p][q] = vect[x][y];
                    q++;
                }
                p++;
            }
            solution[i][j] = pow(-1, i + j) * getDeterminant(subVect);
        }
    }
    return solution;
}
std::vector<std::vector<double>> getInverse(const std::vector<std::vector<double>> vect) {
    if(getDeterminant(vect) == 0)
    {
        throw std::runtime_error("Determinant is 0");
    }

    double d = 1.0/getDeterminant(vect);
    std::vector<std::vector<double>> solution(vect.size(), std::vector<double> (vect.size()));

    for(size_t i = 0; i < vect.size(); i++) {
        for(size_t j = 0; j < vect.size(); j++) {
            solution[i][j] = vect[i][j];
        }
    }

    solution = getTranspose(getCofactor(solution));

    for(size_t i = 0; i < vect.size(); i++) {
        for(size_t j = 0; j < vect.size(); j++) {
            solution[i][j] *= d;
        }
    }

    return solution;
}
void printMatrix(const std::vector<std::vector<double>> vect) {
        for(std::size_t i = 0; i < vect.size(); i++) {
            for(std::size_t j = 0; j < vect[0].size(); j++) {
                std::cout << std::setw(8) << vect[i][j] << " ";
            }
            std::cout << "\n";
        }
    }

class MultiWeightedLinearRegression
{
    public:
    int num_rows;
    int run_once_WLR = 0;
    double alpha_WLR = 0.9999;
    double ESTIMATED_VALUE = 0;
    double PREV_TARGET_VALUE = 1;
    std::vector<double> PREV_EXPLANATORY_VALUES;
    double curr_timestamp = 0;
    std::vector<std::vector<double>> INVERSE_M;
    std::vector<std::vector<double>> ESTIMATE_B;
    std::vector<std::vector<double>> M;
    std::vector<std::vector<double>> V;

    // Global variables for rolling average observed and predicted values
    double rollingAverageObserved = 0.0;
    double rollingAveragePredicted = 0.0;

    // Global variables for sum of squared differences
    double sumSquaredDiffObserved = 0.0;
    double sumSquaredDiffResiduals = 0.0;
    double residualObserved = 0.0;
    double residualPredicted = 0.0;
    double rSquared = 0.0;

    // Function to calculate the rolling R-squared using a one-pass rolling average with a forgetting factor
    double calculateRollingRSquared(double* observed, double* predicted)
    {
        // Update the rolling averages
        rollingAverageObserved  = alpha_WLR * rollingAverageObserved  + (1.0 - alpha_WLR) * (*observed);
        rollingAveragePredicted = alpha_WLR * rollingAveragePredicted + (1.0 - alpha_WLR) * (*predicted);

        // Calculate the current residuals
        residualObserved  = (*observed)  - rollingAverageObserved;
        residualPredicted = (*predicted) - rollingAveragePredicted;

        // Update the sum of squared differences
        sumSquaredDiffObserved  = alpha_WLR * sumSquaredDiffObserved  + (1.0 - alpha_WLR) * ((*observed) * (*observed));
        sumSquaredDiffResiduals = alpha_WLR * sumSquaredDiffResiduals + (1.0 - alpha_WLR) * (residualObserved * residualObserved);

        // Calculate the R-squared
        rSquared = 1.0 - (sumSquaredDiffResiduals / sumSquaredDiffObserved);
        return rSquared;
    }

    void update_WLR(const std::vector<double>& EXPLANATORY_VALUES, double TARGET_VALUE)
    {
        try
        {
            if (run_once_WLR == 1)
            {
                if (PREV_EXPLANATORY_VALUES != EXPLANATORY_VALUES || PREV_TARGET_VALUE != TARGET_VALUE)
                {
                    //int num_explanatory_values = EXPLANATORY_VALUES.size();
                    //int num_rows = num_explanatory_values + 1;


                    // Update M matrix
                    for (int i = 0; i < num_rows; i++) {
                        for (int j = 0; j < num_rows; j++) {
                            if (i == 0 && j == 0)
                                M[i][j] = (1 * (1 - alpha_WLR)) + (M[i][j] * alpha_WLR);
                            else if (i == 0)
                                M[i][j] = (EXPLANATORY_VALUES[j - 1] * (1 - alpha_WLR)) + (M[i][j] * alpha_WLR);
                            else if (j == 0)
                                M[i][j] = (EXPLANATORY_VALUES[i - 1] * (1 - alpha_WLR)) + (M[i][j] * alpha_WLR);
                            else
                                M[i][j] = ((EXPLANATORY_VALUES[i - 1] * EXPLANATORY_VALUES[j - 1]) * (1 - alpha_WLR)) + (M[i][j] * alpha_WLR);
                        }
                    }

                    // Update V matrix
                    for (int i = 0; i < num_rows; i++) {
                        if (i == 0)
                            V[i][0] = (alpha_WLR * V[i][0]) + ((1 - alpha_WLR) * (1 * TARGET_VALUE));
                        else
                            V[i][0] = (alpha_WLR * V[i][0]) + ((1 - alpha_WLR) * (EXPLANATORY_VALUES[i - 1] * TARGET_VALUE));
                    }

                    // Update inverse of M matrix
                    INVERSE_M = getInverse(M);

                    // Update ESTIMATE_B matrix
                    for (int i = 0; i < num_rows; i++) {
                        double sum = 0;
                        for (int j = 0; j < num_rows; j++) {
                            sum += INVERSE_M[i][j] * V[j][0];
                        }
                        ESTIMATE_B[i][0] = sum;
                    }

                    // Calculate estimated value
                    ESTIMATED_VALUE = 0;
                    for (int i = 0; i < num_rows; i++) {
                        if (i == 0)
                            ESTIMATED_VALUE += (ESTIMATE_B[i][0] * 1);
                        else
                            ESTIMATED_VALUE += (EXPLANATORY_VALUES[i - 1] * ESTIMATE_B[i][0]);
                    }

                    PREV_EXPLANATORY_VALUES = EXPLANATORY_VALUES;
                    PREV_TARGET_VALUE = TARGET_VALUE;
                }
            }
            else
            {
                for (double val : EXPLANATORY_VALUES)
                {
                    if (val == 0)
                    {
                        return;
                    }
                }

                for (double val : EXPLANATORY_VALUES)
                {
                    if (std::isnan(val))
                    {
                        return;
                    }
                }

                if (std::isnan(TARGET_VALUE))
                {
                    return;
                }
                if (TARGET_VALUE == 0)
                {
                    return;
                }


                //int num_explanatory_values = EXPLANATORY_VALUES.size();
                //int num_rows = num_explanatory_values + 1;

                M.resize(num_rows, std::vector<double>(num_rows));
                V.resize(num_rows, std::vector<double>(1));
                INVERSE_M.resize(num_rows, std::vector<double>(num_rows));
                ESTIMATE_B.resize(num_rows, std::vector<double>(1));

                // Initialize M matrix
                for (int i = 0; i < num_rows; i++) {
                    for (int j = 0; j < num_rows; j++) {
                        if (i == 0 && j == 0)
                            M[i][j] = 1;
                        else if (i == 0)
                            M[i][j] = EXPLANATORY_VALUES[j - 1];
                        else if (j == 0)
                            M[i][j] = EXPLANATORY_VALUES[i - 1];
                        else
                            M[i][j] = EXPLANATORY_VALUES[i - 1] * EXPLANATORY_VALUES[j - 1];
                    }
                }

                // Initialize V matrix
                for (int i = 0; i < num_rows; i++) {
                    if (i == 0)
                        V[i][0] = 1 * TARGET_VALUE;
                    else
                        V[i][0] = EXPLANATORY_VALUES[i - 1] * TARGET_VALUE;
                }

                run_once_WLR = 1;
            }

        }
        catch(std::exception const& e)
        {
            //cout <<META_ptr->ESTIMATED_VALUE<<","<<RAF_ALPHA->z_score<<","<<VAMP_SIG<<","<<(SHM_PUB_BIN_BTCTUSD_ptr->MO_Intensity_UP - SHM_PUB_BIN_BTCTUSD_ptr->MO_Intensity_DN)<<endl;
            //std::cerr << "Error: " << e.what() << std::endl;

            for (int i = 0; i < EXPLANATORY_VALUES.size(); i++)
            {
                //std::cout << EXPLANATORY_VALUES[i] << ",";
            }

            //std::cout << TARGET_VALUE << std::endl;
            //return EXIT_FAILURE;
        }
    }

    void calc_WLR(const std::vector<double>& EXPLANATORY_VALUES)
    {
        if (run_once_WLR == 1)
        {
            //int num_explanatory_values = EXPLANATORY_VALUES.size();
            //int num_rows = num_explanatory_values + 1;

            // Calculate estimated value
            ESTIMATED_VALUE = 0;
            for (int i = 0; i < num_rows; i++)
            {
                if (i == 0)
                    ESTIMATED_VALUE += (ESTIMATE_B[i][0] * 1);
                else
                    ESTIMATED_VALUE += (EXPLANATORY_VALUES[i - 1] * ESTIMATE_B[i][0]);
            }
        }
    }

    void setAlphaByNbOfUpdates(double nbUpdates)
    {

        this->alpha_WLR = 1.00 - (1.00 / nbUpdates);
    }

    MultiWeightedLinearRegression(int num_explanatory_values, double alpha = 0.9999)
    {
        num_rows = num_explanatory_values + 1;
        M.resize(num_rows, std::vector<double>(num_rows));
        V.resize(num_rows, std::vector<double>(1));
        INVERSE_M.resize(num_rows, std::vector<double>(num_rows));
        ESTIMATE_B.resize(num_rows, std::vector<double>(1));

        this->alpha_WLR = alpha;
    }


};
