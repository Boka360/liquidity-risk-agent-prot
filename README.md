Liquidity Risk Analysis Agent Prototype
Overview
This prototype uses CrewAI to build an AI agent system that ingests financial data from Excel, PDF, TXT, or Word files, computes liquidity risk metrics (e.g., current ratio, quick ratio), and generates a Markdown report with charts. The UI is built with Streamlit.
Setup

Install Dependencies:pip install -r requirements.txt


Set Environment Variables:Create a .env file with:OPENAI_API_KEY=your_key_here


Run the App:streamlit run app.py


Usage:
Open the Streamlit UI (usually at http://localhost:8501).
Upload files (Excel with tabs like 'balance_sheet', PDF, TXT, or Word).
Click "Analyze" to generate a report.



Project Structure

src/crew.py: Defines CrewAI agents and tasks.
src/tools.py: Custom tools for data ingestion, analysis, and reporting.
app.py: Streamlit UI for file uploads and report display.
requirements.txt: Dependencies.
Knowledge/: Unused in this prototype but can store data for future extensions.

Notes

Customize src/tools.py for specific liquidity metrics or data formats.
Test with sample Excel files (e.g., tabs with 'Current Assets', 'Current Liabilities').
Current limitations: Basic error handling, assumes structured data, may need advanced parsing for complex PDFs.

Limitations

Accuracy depends on data quality and LLM prompting—review outputs.
Add error handling for production use.
GPT-4o-mini is used for cost-efficiency; adjust in crew.py for other models.
