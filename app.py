import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

# Define the core investment model class
class StockInvestmentModel:
    """
    A class to model and assess stock investment risk based on various factors.
    """
    def __init__(self, weights=None, confidence_weights=None):
        # Risk factor weights (sum to 1)
        self.weights = weights or {
            'de_ratio': 0.25,
            'beta': 0.25,
            'earnings_volatility': 0.2,
            'sector_risk': 0.15,
            'macro_risk': 0.15
        }
        # Confidence weights (industry typical: data quality > sentiment)
        self.confidence_weights = confidence_weights or {
            'sentiment_confidence': 0.4,
            'risk_data_confidence': 0.6
        }

    # Normalization functions for various risk factors
    def normalize_de_ratio(self, de_ratio):
        """Normalizes Debt-to-Equity ratio to a score from 0.0 to 1.0."""
        if de_ratio > 2: return 1.0
        elif de_ratio >= 1: return 0.5
        else: return 0.0

    def normalize_beta(self, beta):
        """Normalizes Beta to a score from 0.0 to 1.0."""
        if beta > 1.5: return 1.0
        elif beta >= 1.0: return 0.5
        else: return 0.0

    def normalize_earnings_vol(self, vol):
        """Normalizes earnings volatility to a score from 0.0 to 1.0."""
        if vol > 0.3: return 1.0
        elif vol > 0.15: return 0.5
        else: return 0.0

    def normalize_sector_risk(self, risk_level):
        """Normalizes sector risk string to a score from 0.0 to 1.0."""
        mapping = {'high': 1.0, 'medium': 0.5, 'low': 0.0}
        return mapping.get(risk_level.lower(), 0.5)

    def normalize_macro_risk(self, risk_score):
        """Normalizes a macro risk score to a value between 0.0 and 1.0."""
        return min(max(risk_score, 0), 1)

    def calculate_risk_score(self, de_ratio, beta, earnings_volatility, sector_risk, macro_risk):
        """Calculates a weighted risk score based on normalized factors."""
        s_de = self.normalize_de_ratio(de_ratio)
        s_beta = self.normalize_beta(beta)
        s_earnings = self.normalize_earnings_vol(earnings_volatility)
        s_sector = self.normalize_sector_risk(sector_risk)
        s_macro = self.normalize_macro_risk(macro_risk)
        return (
            s_de * self.weights['de_ratio'] +
            s_beta * self.weights['beta'] +
            s_earnings * self.weights['earnings_volatility'] +
            s_sector * self.weights['sector_risk'] +
            s_macro * self.weights['macro_risk']
        )

    def risk_level(self, risk_score):
        """Categorizes a risk score into Low, Medium, or High."""
        if risk_score < 0.33: return 'Low'
        elif risk_score < 0.66: return 'Medium'
        else: return 'High'

    def investment_assessment(self, sentiment_score, risk_score):
        """Provides a qualitative investment assessment and recommended action."""
        risk_lvl = self.risk_level(risk_score)
        if sentiment_score > 0.3:
            if risk_lvl == 'Low': return 'Low Risk Investment', 'High', 'Buy'
            elif risk_lvl == 'Medium': return 'Medium Risk Investment', 'Medium', 'Hold'
            else: return 'Medium to High Risk Investment', 'Medium', 'Hold'
        elif -0.3 <= sentiment_score <= 0.3:
            if risk_lvl == 'Low': return 'Medium Risk Investment', 'Medium', 'Hold'
            elif risk_lvl == 'Medium': return 'Medium Risk Investment', 'Medium', 'Hold'
            else: return 'High Risk Investment', 'High', 'Sell'
        else:
            if risk_lvl == 'Low': return 'Medium to High Risk Investment', 'Medium', 'Hold/Sell (cautious)'
            elif risk_lvl == 'Medium': return 'High Risk Investment', 'High', 'Sell'
            else: return 'Very High Risk Investment', 'Very High', 'Sell'

    def calculate_confidence(self, sentiment_confidence, risk_data_confidence):
        """Calculates a weighted confidence score based on data quality."""
        cw = self.confidence_weights
        confidence_score = (
            sentiment_confidence * cw['sentiment_confidence'] +
            risk_data_confidence * cw['risk_data_confidence']
        )
        return confidence_score * 100  # Convert to percentage

# Function to assess stock
def assess_stock_without_historical(
    sentiment_score, de_ratio, beta, earnings_volatility, sector_risk,
    macro_risk, sentiment_confidence, risk_data_confidence):
    """
    Assess stock - Excludes historical accuracy from confidence.
    """
    # Use industry standard confidence weights
    confidence_weights = {
        'sentiment_confidence': 0.4,  # Typical: 30-40% weight for news sentiment
        'risk_data_confidence': 0.6    # Typical: 60-70% weight for financial data quality
    }
    model = StockInvestmentModel(confidence_weights=confidence_weights)
    risk_score = model.calculate_risk_score(de_ratio, beta, earnings_volatility, sector_risk, macro_risk)
    assessment, conf_label, action = model.investment_assessment(sentiment_score, risk_score)
    confidence_percent = model.calculate_confidence(sentiment_confidence, risk_data_confidence)
    return {
        'sentiment_score': sentiment_score,
        'risk_score': risk_score,
        'risk_level': model.risk_level(risk_score),
        'assessment': assessment,
        'confidence_percent': confidence_percent,
        'recommended_action': action,
        'de_ratio': de_ratio,
        'beta': beta,
        'earnings_volatility': earnings_volatility,
    }


# Functions to fetch data using yfinance
def calculate_earnings_volatility(ticker_symbol):
    """
    Fetches quarterly financial data and calculates earnings volatility.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        financials_df = ticker.quarterly_financials.T
        if financials_df.empty or 'Net Income' not in financials_df.columns:
            st.warning(f"No 'Net Income' data found for {ticker_symbol}.")
            return None
        earnings = financials_df['Net Income'].dropna()
        if earnings.empty:
            st.warning(f"No 'Net Income' data found for {ticker_symbol}.")
            return None
        std_dev = np.std(earnings.values)
        mean_earnings = np.mean(earnings.values)
        if mean_earnings == 0:
            st.warning(f"Average earnings for {ticker_symbol} is zero. Cannot calculate volatility.")
            return None
        coefficient_of_variation = (std_dev / abs(mean_earnings))
        return coefficient_of_variation
    except Exception as e:
        st.error(f"An error occurred while fetching earnings data: {e}")
        return None

def calculate_debt_to_equity(ticker_symbol):
    """
    Fetches balance sheet data and calculates the Debt-to-Equity ratio.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        balance_sheet = ticker.balance_sheet
        if balance_sheet.empty:
            st.warning(f"No balance sheet data found for {ticker_symbol}.")
            return None
        total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
        total_equity = balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0]
        if total_liabilities is None or total_equity is None or total_equity == 0:
            st.warning(f"Required balance sheet data not found or Total Equity is zero for {ticker_symbol}.")
            return None
        debt_to_equity_ratio = total_liabilities / total_equity
        return debt_to_equity_ratio
    except KeyError:
        st.error(f"Required balance sheet data not found for {ticker_symbol}. Check if the company has a valid balance sheet on Yahoo Finance.")
        return None
    except Exception as e:
        st.error(f"An error occurred while fetching D/E ratio: {e}")
        return None
        
# --- New VaR/CVaR Calculator Class ---
class VaRCalculator:
    """
    Calculates Value at Risk (VaR) and Conditional VaR (CVaR)
    based on historical returns.
    """
    def __init__(self, historical_returns):
        self.returns = historical_returns

    def calculate_var(self, confidence_level=0.95):
        """
        Calculates VaR at a given confidence level.
        The VaR is the negative of the specified percentile of the returns distribution.
        """
        alpha = 1 - confidence_level
        var_value = np.percentile(self.returns, alpha * 100)
        return -var_value

    def calculate_cvar(self, confidence_level=0.95):
        """
        Calculates CVaR (Expected Shortfall) at a given confidence level.
        CVaR is the average of all losses that are worse than the VaR.
        """
        var_value = self.calculate_var(confidence_level)
        tail_losses = self.returns[self.returns <= -var_value]
        if tail_losses.empty:
            return 0  # Handle case where no losses exceed VaR
        cvar_value = -np.mean(tail_losses)
        return cvar_value


# --- Data and Mapping ---
# Beta values and company lists, based on your provided comments
beta = {
    'Metal': {'beta': 2.2, 'companies': {
        'JSW Steel': 'JSWSTEEL.NS', 'Tata Steel': 'TATASTEEL.NS', 'Vedanta': 'VEDL.NS',
        'Hindalco Industries': 'HINDALCO.NS', 'Hindustan Zinc': 'HINDZINC.NS',
        'SAIL': 'SAIL.NS', 'NMDC': 'NMDC.NS'
    }},
    'Automotive': {'beta': 1.8, 'companies': {
        'Tata Motors': 'TATAMOTORS.NS', 'Bajaj Auto': 'BAJAJ-AUTO.NS', 'Eicher Motors': 'EICHERMOT.NS',
        'Hero MotoCorp': 'HEROMOTOCO.NS', 'Mahindra & Mahindra': 'M&M.NS',
        'Ashok Leyland': 'ASHOKLEY.NS', 'Maruti Suzuki India': 'MARUTI.NS', 'TVS Motor': 'TVSMOTOR.NS'
    }},
    'Financial Services': {'beta': 1.6, 'companies': {
        'Bajaj Finance': 'BAJFINANCE.NS', 'Muthoot Finance': 'MUTHOOTFIN.NS', 'Aditya Birla Capital': 'ABCAPITAL.NS',
        'HDFC Bank': 'HDFCBANK.NS', 'ICICI Bank': 'ICICIBANK.NS', 'Shriram Finance': 'SHRIRAMFIN.NS',
        'State Bank of India': 'SBIN.NS', 'LIC': 'LICI.NS'
    }},
    'Energy': {'beta': 1.4, 'companies': {
        'JSW Energy': 'JSWENERGY.NS', 'Tata Power': 'TATAPOWER.NS', 'Adani Power': 'ADANIPOWER.NS',
        'Reliance Power': 'RELPOWER.NS', 'Power Grid Corp': 'POWERGRID.NS',
        'Indian Oil Corp': 'IOC.NS', 'Bharat Petroleum Corp': 'BPCL.NS', 'NTPC': 'NTPC.NS'
    }},
    'Technology': {'beta': 1.1, 'companies': {
        'TCS': 'TCS.NS', 'Wipro': 'WIPRO.NS', 'Infosys': 'INFY.NS',
        'Tech Mahindra': 'TECHM.NS', 'HCL Technologies': 'HCLTECH.NS',
        'Persistent Systems': 'PERSISTENT.NS', 'Larsen & Toubro Infotech': 'LTIM.NS', 'Tata Elxsi': 'TATAELXSI.NS'
    }},
    'Consumers': {'beta': 0.7, 'companies': {
        'Blue Star': 'BLUESTARCO.NS', 'Voltas': 'VOLTAS.NS', 'Crompton Greaves': 'CROMPTON.NS',
        'Havells India': 'HAVELLS.NS', 'Bajaj Electricals': 'BAJAJELEC.NS', 'Whirlpool of India': 'WHIRLPOOL.NS',
        'Titan Company': 'TITAN.NS', 'Asian Paints': 'ASIANPAINT.NS'
    }},
    'FMCG': {'beta': 0.8, 'companies': {
        'Dabur': 'DABUR.NS', 'Godrej Consumer': 'GODREJCP.NS', 'Britannia': 'BRITANNIA.NS',
        'ITC': 'ITC.NS', 'Nestle India': 'NESTLEIND.NS', 'Marico': 'MARICO.NS',
        'Tata Consumer Products': 'TATACONSUM.NS', 'Colgate-Palmolive': 'COLPAL.NS'
    }},
    'Pharma': {'beta': 1.05, 'companies': {
        'Cipla': 'CIPLA.NS', 'Dr. Reddys Labs': 'DRREDDY.NS', 'Mankind Pharma': 'MANKIND.NS',
        'Sun Pharmaceutical': 'SUNPHARMA.NS', 'Torrent Pharma': 'TORNTPHARM.NS',
        'Lupin': 'LUPIN.NS', 'Zydus Lifesciences': 'ZYDUSLIFE.NS', 'Biocon': 'BIOCON.NS'
    }}
}

# Streamlit App UI
st.set_page_config(page_title="Stock Assessment App", layout="centered")
st.title("📈 AI-Powered Stock Investment Assessment")
st.markdown("---")
st.markdown("""
Welcome to the stock assessment app! This tool helps you evaluate the risk and potential of an investment based on real-time financial data.
Simply select a sector and a company, and the app will do the rest.
""")

st.markdown("---")
st.header("1. Select Your Investment")

# Create a selectbox for the sector
sector = st.selectbox(
    "Select a Sector:",
    options=list(beta.keys())
)

# Get the list of companies for the selected sector and create a selectbox
company_list = list(beta[sector]['companies'].keys())
selected_company = st.selectbox(
    "Select a Company:",
    options=company_list
)

# Get the corresponding ticker symbol from the mapping
ticker_symbol = beta[sector]['companies'][selected_company]

# Allow the user to input the investment amount
investment_amount = st.number_input(
    "Enter Investment Amount (in INR):",
    min_value=1000,
    value=10000,
    step=1000
)

# Create a button to trigger the assessment
if st.button("Assess Investment"):
    st.markdown("---")
    st.header("2. Performing Assessment")
    st.info(f"Fetching data for **{selected_company} ({ticker_symbol})**...")
    
    # Get the static beta value from our predefined dictionary
    static_beta = beta[sector]['beta']
    
    # Fetch dynamic data from yfinance
    try:
        de_ratio = calculate_debt_to_equity(ticker_symbol)
        earnings_volatility = calculate_earnings_volatility(ticker_symbol)

        # --- New VaR/CVaR Calculation ---
        ticker_data = yf.Ticker(ticker_symbol)
        hist = ticker_data.history(period="1y")
        if hist.empty:
            st.error("Could not fetch historical data for VaR/CVaR calculations.")
            st.stop()
            
        # Calculate daily percentage returns
        daily_returns = hist['Close'].pct_change().dropna()
        if daily_returns.empty:
            st.error("Not enough historical data to calculate returns for VaR/CVaR.")
            st.stop()

        # Instantiate the calculator and compute VaR and CVaR
        confidence_level = 0.95
        var_calc = VaRCalculator(daily_returns)
        var = var_calc.calculate_var(confidence_level)
        cvar = var_calc.calculate_cvar(confidence_level)

        # Check if data fetching was successful
        if de_ratio is None or earnings_volatility is None:
            st.error("Could not complete the assessment due to missing financial data.")
        else:
            # --- Assumptions for the assessment ---
            sentiment_score = 0.2
            sector_risk = 'medium'
            macro_risk = 0.4
            sentiment_confidence = 0.9
            risk_data_confidence = 0.85
            
            # Run the assessment model
            result = assess_stock_without_historical(
                sentiment_score=sentiment_score,
                de_ratio=de_ratio,
                beta=static_beta,
                earnings_volatility=earnings_volatility,
                sector_risk=sector_risk,
                macro_risk=macro_risk,
                sentiment_confidence=sentiment_confidence,
                risk_data_confidence=risk_data_confidence
            )
            
            # Display results
            st.markdown("---")
            st.header("3. Investment Analysis Results")
            
            # Use columns to present key metrics side-by-side
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Risk Score", value=f"{result['risk_score']:.2f}")
            with col2:
                st.metric(label="Risk Level", value=result['risk_level'])
            with col3:
                st.metric(label="Confidence Level", value=f"{result['confidence_percent']:.0f}%")
            
            # Determine the color and icon for the recommended action
            action = result['recommended_action'].upper()
            if 'BUY' in action:
                st.success(f"**Recommended Action: {action}**")
            elif 'HOLD' in action:
                st.warning(f"**Recommended Action: {action}**")
            else:
                st.error(f"**Recommended Action: {action}**")
                
            st.write(f"**Investment Assessment:** {result['assessment']}")
            st.write(f"Based on the data, the calculated Debt-to-Equity ratio is **{result['de_ratio']:.2f}** and the historical earnings volatility is **{result['earnings_volatility']:.2f}**.")
            st.write(f"The sector Beta for {sector} is **{result['beta']}**, indicating its volatility relative to the market.")

            # --- Display VaR and CVaR in a new section ---
            st.markdown("---")
            st.header("4. Statistical Risk Metrics (VaR & CVaR)")
            st.markdown(f"**Confidence Level:** {confidence_level*100:.0f}% over a 1-day horizon")
            
            # Calculate and display the potential loss in rupees based on the investment amount
            var_rupees = var * investment_amount
            cvar_rupees = cvar * investment_amount

            var_col, cvar_col = st.columns(2)
            with var_col:
                st.info(f"**VaR (Value at Risk)**: A potential daily loss of up to **₹{var_rupees:,.2f}** is expected in 5% of trading days.")
                st.info(f"**VaR as Percentage**: {var*100:.2f}%")
            with cvar_col:
                st.warning(f"**CVaR (Conditional VaR)**: The average loss on the worst 5% of trading days is **₹{cvar_rupees:,.2f}**.")
                st.warning(f"**CVaR as Percentage**: {cvar*100:.2f}%")
            
            st.markdown("""
            - **VaR** quantifies the maximum expected loss with a specified confidence level.
            - **CVaR** (or Expected Shortfall) measures the average loss in the worst-case scenarios, providing a more conservative risk estimate.
            """)
            
    except Exception as e:
        st.error(f"An unexpected error occurred during the assessment: {e}")