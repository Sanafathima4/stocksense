from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Constants
TICKER_PATTERN = r'\$?([A-Z]{1,5})\b'
TECHNICAL_KEYWORDS = {
    'programming', 'code', 'algorithm', 'software', 'develop', 'debug', 'database',
    'python', 'java', 'javascript', 'c++', 'html', 'css', 'react', 'node', 'sql',
    'git', 'docker', 'kubernetes', 'api', 'framework', 'backend', 'frontend',
    'machine learning', 'ai', 'neural network', 'cloud', 'aws', 'azure', 'server',
    'compiler', 'debugging', 'optimization', 'cybersecurity', 'database', 'linux',
    'windows', 'macos', 'terminal', 'scripting', 'oop', 'functional programming',
    'devops', 'rest', 'graphql', 'microservices', 'agile', 'scrum', 'testing'
}
STOCK_KEYWORDS = {
    'stock', 'share', 'price', 'market', 'nasdaq', 'nyse', 'dividend', 'etf',
    'portfolio', 'valuation', 'pe ratio', 'eps', 'ipo', 'merger', 'acquisition',
    'earnings', 'revenue', 'profit', 'loss', 'capital', 'asset', 'crypto',
    'bitcoin', 'ethereum', 'ticker', 'exchange', 'index', 's&p', 'dow jones',
    'volume', 'short selling', 'option', 'future', 'bull', 'bear', 'roe', 'roi',
    'market cap', 'balance sheet', 'income statement', 'cash flow', 'ebitda',
    'valuation', 'technical analysis', 'fundamental analysis', 'moving average',
    'rsi', 'macd', 'volatility', 'blue chip', 'growth stock', 'value investing'
}

# Load and process knowledge base
loader = TextLoader("Knowledgebase.txt")  # Updated filename
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# Create vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    system_instruction="""You are a technical and financial analyst. Use the provided context and live data to answer questions. 
    Combine knowledge base information with real-time market data. Be clear about data sources. 
    If information conflicts, prioritize recent market data."""
)

chat_history = []

def is_relevant(query: str) -> bool:
    """Check if query contains technical or stock-related terms"""
    query_lower = query.lower()
    if any(term in query_lower for term in TECHNICAL_KEYWORDS): return True
    if any(term in query_lower for term in STOCK_KEYWORDS): return True
    if re.search(TICKER_PATTERN, query): return True
    return False

def fetch_financial_data(ticker: str, query: str) -> str:
    """Fetch and process financial data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty: return ""
        
        # Calculate indicators
        hist['SMA_50'] = hist['Close'].rolling(50).mean()
        hist['SMA_200'] = hist['Close'].rolling(200).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        
        latest = hist.iloc[-1]
        data_str = f"""
        {ticker.upper()} Technical Snapshot:
        - Price: ${latest['Close']:.2f}
        - 50-Day SMA: ${latest['SMA_50']:.2f}
        - 200-Day SMA: ${latest['SMA_200']:.2f}
        - RSI (14): {latest['RSI']:.1f}
        - Volume: {latest['Volume']:,}
        - 52W Range: ${hist['Low'].min():.2f}-${hist['High'].max():.2f}
        """    
        if "macd" in query.lower():
            macd, signal = calculate_macd(hist['Close'])
            data_str += f"\nMACD: {macd[-1]:.2f}\nSignal: {signal[-1]:.2f}"
            
        if "bollinger" in query.lower():
            upper, lower = calculate_bollinger_bands(hist['Close'])
            data_str += f"\nBollinger Bands:\n- Upper: ${upper[-1]:.2f}\n- Lower: ${lower[-1]:.2f}"
            
        return data_str
        
    except Exception as e:
        return f"Data Error: {str(e)}"

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return sma + (std * num_std), sma - (std * num_std)

# FastAPI application
from fastapi import FastAPI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (replace with specific origins in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    user_input = request.query
    
    if not is_relevant(user_input):
        raise HTTPException(status_code=400, detail="I specialize in technical and financial topics. Please ask relevant questions.")
    
    # Retrieve knowledge base context
    try:
        docs = retriever.invoke(user_input)
        context = "\n".join([d.page_content for d in docs])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval Error: {str(e)}")
    
    # Add live market data
    ticker_match = re.search(TICKER_PATTERN, user_input, re.IGNORECASE)
    live_data = ""
    if ticker_match:
        ticker = ticker_match.group(1)
        live_data = fetch_financial_data(ticker, user_input)
        context = f"Knowledge Base:\n{context}\n\nLive Data:\n{live_data}"
    
    # Generate response
    try:
        response = model.invoke([
            *chat_history,
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {user_input}")
        ])
        
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response.content)
        ])
        
        return {"response": response.content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation Error: {str(e)}")
