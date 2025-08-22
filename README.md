# Stock Investment Assessment App 📈

An AI-powered Streamlit application that helps investors evaluate stock investment risks and opportunities based on real-time financial data and comprehensive risk analysis.

## Features

- **Real-time Data Analysis**: Fetches live financial data using Yahoo Finance API
- **Multi-factor Risk Assessment**: Analyzes multiple risk factors including:
  - Debt-to-Equity ratio
  - Beta (volatility)
  - Earnings volatility
  - Sector risk
  - Macro risk factors
- **Sector-based Analysis**: Pre-categorized companies across 8 major sectors
- **Investment Recommendations**: Clear BUY/HOLD/SELL recommendations
- **Confidence Scoring**: Assessment reliability based on data quality

## Supported Sectors & Companies

The app covers 8 major sectors with prominent Indian companies:

- **Technology** (Beta: 1.1): TCS, Infosys, Wipro, HCL Tech, etc.
- **Financial Services** (Beta: 1.6): HDFC Bank, ICICI Bank, Bajaj Finance, etc.
- **FMCG** (Beta: 0.8): ITC, Nestle India, Britannia, Dabur, etc.
- **Pharma** (Beta: 1.05): Sun Pharma, Dr. Reddy's, Cipla, Lupin, etc.
- **Automotive** (Beta: 1.8): Maruti Suzuki, Tata Motors, Bajaj Auto, etc.
- **Energy** (Beta: 1.4): Reliance, NTPC, BPCL, IOC, etc.
- **Metal** (Beta: 2.2): Tata Steel, JSW Steel, Vedanta, etc.
- **Consumer Goods** (Beta: 0.7): Asian Paints, Titan, Havells, etc.

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository**

2. **Install required packages**

3. **Run the application**

The app will open in your default web browser at `http://localhost:8501`

## How to Use

1. **Select Investment Parameters**
   - Choose a sector from the dropdown
   - Select a company within that sector
   - Enter your investment amount (for reference)

2. **Perform Assessment**
   - Click "Assess Investment" button
   - The app fetches real-time financial data
   - Analysis is performed using the investment model

3. **Review Results**
   - View risk score, risk level, and confidence percentage
   - Get clear investment recommendations
   - Understand key financial metrics

## Risk Assessment Model

The app uses a sophisticated weighted scoring system:

### Risk Factors & Weights
- **Debt-to-Equity Ratio** (25%): Financial leverage assessment
- **Beta** (25%): Market volatility relative to benchmark
- **Earnings Volatility** (20%): Historical earnings consistency
- **Sector Risk** (15%): Industry-specific risk factors
- **Macro Risk** (15%): Overall market conditions

### Risk Categorization
- **Low Risk** (< 0.33): Generally safe investments
- **Medium Risk** (0.33 - 0.66): Moderate risk with balanced potential
- **High Risk** (> 0.66): Higher volatility but potential for greater returns

## Technical Architecture

### Core Components

1. **StockInvestmentModel Class**
   - Handles risk calculation and normalization
   - Provides investment assessments and recommendations
   - Calculates confidence scores

2. **Data Fetching Functions**
   - `calculate_earnings_volatility()`: Quarterly earnings analysis
   - `calculate_debt_to_equity()`: Balance sheet analysis
   - Real-time data via yfinance API

3. **Streamlit Interface**
   - Interactive sector/company selection
   - Real-time assessment results
   - Visual metrics and recommendations

### Data Sources
- **Yahoo Finance API**: Real-time financial data
- **Predefined Sector Data**: Beta values and company mappings
- **Financial Statements**: Balance sheets and income statements

## Example Output

## Risk Score: 0.45
## Risk Level: Medium
## Confidence Level: 87%
## Recommended Action: HOLD