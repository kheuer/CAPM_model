# Portfolio Optimization Web App

This is a web application for portfolio optimization. It provides the user with the optimal portfolio given a set of assets, a risk free rate and constraints regarding the size of each position.

## CAPM
The Capital Asset Pricing Model (CAPM) is a financial model used to calculate the expected return of an investment based on its level of risk. It was developed by William Sharpe, John Lintner, and Jan Mossin in the 1960s.

The model is based on the idea that the expected return of an asset is a function of its beta (β), which measures the asset's volatility in relation to the market as a whole. The beta is used to adjust the expected return of the asset for its level of risk.

The CAPM formula is as follows:

Expected Return = Risk-free Rate + Beta × (Market Return - Risk-free Rate)

Where:

- Risk-free Rate: The return on a risk-free investment such as a government bond, which is considered to have no risk.
- Beta: A measure of an asset's volatility in relation to the market as a whole.
- Market Return: The average return of the stock market as a whole.

The CAPM assumes that investors are rational and risk-averse, and that they require compensation for taking on additional risk. The model also assumes that all investors have access to the same information and that the market is efficient, meaning that prices reflect all available information.

In practice, the CAPM is often used as a benchmark for evaluating the performance of investment portfolios. A portfolio with a higher expected return than the CAPM model predicts is considered to have outperformed the market, while a portfolio with a lower expected return is considered to have underperformed.

## Technical Implementation
The Minimum Variance Portfolio (MVP) is the portfolio that has the minimum variance among all possible portfolios. It is important because it represents the most diversified portfolio for a given set of assets. 
The Maximum Sharpe Ratio Portfolio (MSRP) is the portfolio that has the highest ratio of excess returns to volatility. It represents the optimal portfolio for a given level of risk.

Any Selection from the Capital Market Line or anywhere else on the possibility frontier can then be selected on the main screen and corresponding position sizes can be obtained as seen in the screenshots below.
![Portfolio Selection](https://raw.githubusercontent.com/kheuer/CAPM_model/master/main_site.png)


![Portfolio Weighs](https://raw.githubusercontent.com/kheuer/CAPM_model/master/portfolio_view.png)
