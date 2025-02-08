from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
load_dotenv()

## Web serach Agent
web_search_agent = Agent(
    name = "Web search agent",
    role = "Search the web for information",
    model = Groq(id="deepseek-r1-distill-llama-70b"),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tool_calls = True, 
    markdown = True,
    add_datetime_to_instructions=True,
    # debug_mode=True,
)

## Financial agent
finance_agent = Agent(
    name = "Finance Ai Agent", 
    role = "Financial agent",
    model = Groq(id = "deepseek-r1-distill-llama-70b"),
    tools = [
        YFinanceTools(stock_price=True, analyst_recommendations=True, 
                      stock_fundamentals=True,
                      company_news = True)],
    instructions = ["This is a financial analysis bot", "Use tables to display the data"],
    show_tool_calls = True, 
    markdown = True,
    add_datetime_to_instructions=True,
)

multi_ai_agent = Agent(
    team = [web_search_agent, finance_agent],
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    instructions = ["Always include sources", "Use tables to display the data", "This is a financial analysis bot", "Provide Financial information only"],
    show_tool_calls = True, 
    markdown = True,
)

multi_ai_agent.print_response("provide NVDiA price history", stream=True)