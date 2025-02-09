from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
import phi
from phi.playground import Playground, serve_playground_app

# Load environment variables
load_dotenv()
phi.api = os.getenv("PHI_API_KEY")

# Web Search Agent (unchanged)
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
    add_datetime_to_instructions=True,
)

# Hybrid Sentiment and Fundamental Analysis Bot
hybrid_analysis_bot = Agent(
    name="Hybrid Analysis Bot",
    role="Perform sentiment analysis on text and fundamental analysis on stocks",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[
        YFinanceTools(
            stock_price=True,
            stock_fundamentals=True,
            company_news=True
        )
    ],
    instructions=[
        "You are a hybrid analysis expert capable of performing sentiment analysis on text and fundamental analysis on stocks",
        "If the input is text, perform sentiment analysis and provide:",
        "1. The sentiment of the text (positive, negative, or neutral).",
        "2. A confidence score for the sentiment analysis (e.g., 85% positive).",
        "3. Key phrases or words that influenced the sentiment analysis.",
        "4. A brief explanation of why the sentiment was classified as such.",
        "If the input is a stock ticker name, perform fundamental analysis and provide:",
        "1. Current price and price change (in percentage).",
        "2. Market cap, volume, and other key metrics.",
        "3. Recent news related to the stock ",
        "4. Historical performance (if applicable).",
        "Use tables and markdown formatting to present the data clearly.",
        "Always include the source of the data.",
        "If the input is unclear or lacks sufficient information, provide a clear explanation.",
    ],
    show_tool_calls=True,
    markdown=True,
    add_datetime_to_instructions=True,
)

# Create the Playground app
app = Playground(agents=[hybrid_analysis_bot, web_search_agent]).get_app()

# Serve the Playground app
if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
