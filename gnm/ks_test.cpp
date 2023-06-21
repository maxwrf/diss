#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

double ksTest(const std::vector<double> &x, const std::vector<double> &y)
{
    std::vector<double> xSorted = x;
    std::vector<double> ySorted = y;

    std::sort(xSorted.begin(), xSorted.end());
    std::sort(ySorted.begin(), ySorted.end());

    std::vector<double> combinedData;
    combinedData.insert(combinedData.end(), xSorted.begin(), xSorted.end());
    combinedData.insert(combinedData.end(), ySorted.begin(), ySorted.end());

    std::sort(combinedData.begin(), combinedData.end());

    std::vector<double> sortedCombined = combinedData;

    std::sort(sortedCombined.begin(), sortedCombined.end());

    std::vector<double> cdf_x(sortedCombined.size());
    std::vector<double> cdf_y(sortedCombined.size());

    for (size_t i = 0; i < sortedCombined.size(); ++i)
    {
        size_t count_x = std::lower_bound(xSorted.begin(), xSorted.end(), sortedCombined[i]) - xSorted.begin();
        size_t count_y = std::lower_bound(ySorted.begin(), ySorted.end(), sortedCombined[i]) - ySorted.begin();
        cdf_x[i] = static_cast<double>(count_x) / x.size();
        cdf_y[i] = static_cast<double>(count_y) / y.size();
    }

    std::vector<double> diff_cdf(cdf_x.size());
    for (size_t i = 0; i < cdf_x.size(); ++i)
    {
        diff_cdf[i] = std::abs(cdf_x[i] - cdf_y[i]);
    }

    return *std::max_element(diff_cdf.begin(), diff_cdf.end());
}

int main()
{
    std::vector<double> x = {1.2, 2.5, 3.7, 4.1, 5.6};
    std::vector<double> y = {1.0, 2.2, 3.6, 4.4, 5.8};

    double ksStat = ksTest(x, y);

    std::cout << "KS statistic: " << ksStat << std::endl;

    return 0;
}
