import os
from dotenv import load_dotenv
import json
import re
import time
import datetime
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Retail Analytics Bot", layout="wide")

# Fetch API Key
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is missing from your environment. Restart your terminal.")
    st.stop()

# ---------------------------------------------------------
# NEW ARCHITECTURE: MULTI-SESSION STATE MANAGEMENT
# ---------------------------------------------------------
if "sessions" not in st.session_state:
    # A nested dictionary to hold multiple chat instances
    st.session_state.sessions = {"Chat 1": {"chat_history": [], "display_messages": []}}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "Chat 1"
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# Pointer to the active session so we don't have to type this massive string every time
active_session = st.session_state.sessions[st.session_state.current_session_id]

# ---------------------------------------------------------
# CACHED DB & LLM
# ---------------------------------------------------------
@st.cache_resource
def get_database_and_llm():
    db = SQLDatabase.from_uri("sqlite:///retail_sales.db", include_tables=["sales_data"], sample_rows_in_table_info=1)
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, google_api_key=api_key)
    return db, llm

db, llm = get_database_and_llm()

def get_schema(_):
    return db.get_table_info()

def run_query(query: str):
    clean_query = query.replace("```sql", "").replace("```", "").strip()
    try:
        result_string = str(db.run(clean_query))
        if len(result_string) > 5000:
            return result_string[:5000] + "\n... [SYSTEM WARNING: DATA TRUNCATED. Use GROUP BY or LIMIT.]"
        return result_string
    except Exception as e:
        return str(e)

def clean_mashed_text(text):
    if not text: return "No summary provided."
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'\.([a-zA-Z])', r'. \1', text)
    return text

# ---------------------------------------------------------
# LCEL CHAINS
# ---------------------------------------------------------
sql_prompt = ChatPromptTemplate.from_template(
    """Based on the schema below and chat history, write a SQLite query answering the user.
    CRITICAL RULES:
    1. Return ONLY raw SQL. No markdown.
    2. ALWAYS aggregate data (GROUP BY) if a chart/trend is requested.
    3. LIMIT 50 if unaggregated.
    
    Schema: {schema}
    Chat History: {chat_history}
    Question: {question}
    SQL Query:"""
)

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | sql_prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

json_prompt = ChatPromptTemplate.from_template(
    """You are a data analyst assistant. Analyze the user's question and the SQL results.
    You must respond with a STRICT JSON object.

    CRITICAL TYPOGRAPHY RULES:
    - Space between numbers and words. Space after periods.

    JSON SCHEMA:
    1. "text_summary": Explanation of the data. Keep it short.
    2. "needs_chart": Boolean true if visualization is needed.
    3. "chart_type": "bar", "line", "scatter", or "box" (else null).
    4. "x_column": Exact column name for X-axis (or null).
    5. "y_column": Exact column name for Y-axis (or null).
    6. "follow_up_questions": A list of exactly 2 insightful follow-up questions.
    
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    
    Return ONLY valid JSON. No markdown.
    JSON Output:"""
)

json_chain = json_prompt | llm | StrOutputParser()

# ---------------------------------------------------------
# UTILS (Caching, Charting, Jupyter Export)
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_response(user_query, history_str):
    generated_sql = sql_chain.invoke({"question": user_query, "chat_history": history_str})
    sql_result = run_query(generated_sql)
    raw_response = json_chain.invoke({"question": user_query, "query": generated_sql, "result": sql_result})
    return raw_response, generated_sql

def draw_chart(df, chart_type, x_col, y_col, chart_key):
    """Helper function to draw the Plotly charts consistently with unique keys"""
    if chart_type == "bar":
        fig = px.bar(df, x=x_col, y=y_col, title="Analytics Data")
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    elif chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, title="Trend Analysis", markers=True)
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, title="Data Scatter")
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    elif chart_type == "box":
        if x_col and x_col != y_col:
            fig = px.box(df, x=x_col, y=y_col, title="Statistical Distribution")
        else:
            fig = px.box(df, y=y_col, title="Statistical Distribution")
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    else:
        st.dataframe(df)

def generate_notebook_export(messages):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": ["# 📊 Retail Analytics Chat Export"]}]
    for msg in messages:
        role = "🧑‍💻 **User:**" if msg["role"] == "user" else "🤖 **Assistant:**"
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"{role}\n\n{msg['content']}"]})
        if msg.get("has_chart") and msg.get("df") is not None:
            df = msg["df"]
            raw_data = df.to_dict(orient="records")
            code = f"import pandas as pd\nimport plotly.express as px\n\ndata = {raw_data}\ndf = pd.DataFrame(data)\n\n"
            code += f"fig = px.{msg['chart_type']}(df"
            if msg['chart_type'] == 'box' and (not msg['x_col'] or msg['x_col'] == msg['y_col']):
                code += f", y='{msg['y_col']}', title='Statistical Distribution')\n"
            else:
                code += f", x='{msg['x_col']}', y='{msg['y_col']}', title='Data Visualization')\n"
            code += "fig.show()\n"
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code]})
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, indent=2)

# ---------------------------------------------------------
# UI: SIDEBAR WITH CHAT MANAGEMENT
# ---------------------------------------------------------
with st.sidebar:
    
    # The 'Start Fresh' button logic
    if st.button("➕ Start New Chat", use_container_width=True):
        new_id = f"Chat {len(st.session_state.sessions) + 1}"
        st.session_state.sessions[new_id] = {"chat_history": [], "display_messages": []}
        st.session_state.current_session_id = new_id
        st.rerun() # Force UI to immediately refresh and show a blank screen
        
    st.markdown("**Previous Chats:**")
    # Draw a button for every chat in memory
    for session_id in reversed(list(st.session_state.sessions.keys())):
        # Visually highlight the chat we are currently looking at
        button_style = "primary" if session_id == st.session_state.current_session_id else "secondary"
        if st.button(session_id, use_container_width=True, type=button_style, key=f"btn_{session_id}"):
            st.session_state.current_session_id = session_id
            st.rerun() # Jump to the selected chat

    st.markdown("---")
    st.markdown("### Some Suggestions")
    if st.button("📊 Total sales by month"): st.session_state.current_query = "Show me a bar chart of total sales per month"
    if st.button("📈 Sales trend over time"): st.session_state.current_query = "Plot a line chart of sales over the last year"
    
    st.markdown("---")
    st.markdown("### 💾 Export Current Chat")
    
    # Export ONLY the messages from the currently active chat tab
    notebook_payload = generate_notebook_export(active_session["display_messages"])
    st.download_button(
        label="Download Report (.ipynb)", 
        data=notebook_payload, 
        file_name=f"analytics_{st.session_state.current_session_id.replace(' ', '_')}.ipynb", 
        mime="application/x-ipynb+json",
        use_container_width=True
    )

# ---------------------------------------------------------
# UI: MAIN CHAT INTERFACE
# ---------------------------------------------------------
# Dynamically display the name of the current chat you are looking at
st.title(f"📊 {st.session_state.current_session_id}")
# st.markdown("Try not to break the multi-session dictionary.")

# Render historical messages from the ACTIVE session only
for idx, msg in enumerate(active_session["display_messages"]):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Redraw the plot if the memory object has a dataframe
        if msg.get("has_chart") and msg.get("df") is not None:
            # Generate a globally unique key using the session ID and the message index
            unique_chart_key = f"hist_chart_{st.session_state.current_session_id}_{idx}"
            draw_chart(msg["df"], msg.get("chart_type"), msg.get("x_col"), msg.get("y_col"), unique_chart_key)
        if msg.get("follow_ups"):
            cols = st.columns(len(msg["follow_ups"]))
            for i, suggestion in enumerate(msg["follow_ups"]):
                if cols[i].button(suggestion, key=f"fup_{st.session_state.current_session_id}_{idx}_{i}"):
                    st.session_state.current_query = suggestion
                    st.rerun()
        if "meta" in msg:
            st.caption(msg["meta"])

user_input = st.chat_input("Ask a question about your data...")
query_to_run = user_input or st.session_state.current_query

if query_to_run:
    st.session_state.current_query = ""
    
    with st.chat_message("user"):
        st.write(query_to_run)
    active_session["display_messages"].append({"role": "user", "content": query_to_run})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            start_time = time.time()
            try:
                # Grab context from the ACTIVE session
                history_str = "\n".join([f"User: {x['user']}\nAI: {x['ai']}" for x in active_session["chat_history"][-4:]])
                raw_response, generated_sql = get_cached_response(query_to_run, history_str)
                
                clean_json_string = raw_response.replace("```json", "").replace("```", "").strip()
                response_data = json.loads(clean_json_string)
                
                clean_summary = clean_mashed_text(response_data.get("text_summary", "No summary provided."))
                st.write(clean_summary)
                
                has_chart = False
                df = None
                chart_type = response_data.get("chart_type")
                x_col = response_data.get("x_column")
                y_col = response_data.get("y_column")
                
                if response_data.get("needs_chart"):
                    has_chart = True
                    conn = sqlite3.connect("retail_sales.db")
                    df = pd.read_sql_query(generated_sql, conn)
                    conn.close()
                    
                    # Fallback logic in case the LLM hallucinates the column names
                    if x_col not in df.columns: x_col = df.columns[0] if len(df.columns) > 0 else None
                    if y_col not in df.columns: y_col = df.columns[-1] if len(df.columns) > 0 else None
                    
                    # Generate a unique key for the brand new chart
                    new_chart_key = f"live_chart_{st.session_state.current_session_id}_{len(active_session['display_messages'])}"
                    draw_chart(df, chart_type, x_col, y_col, new_chart_key)

                follow_ups = response_data.get("follow_up_questions", [])
                if follow_ups:
                    cols = st.columns(len(follow_ups))
                    for i, suggestion in enumerate(follow_ups):
                        if cols[i].button(suggestion, key=f"new_fup_{st.session_state.current_session_id}_{i}"):
                            st.session_state.current_query = suggestion
                            st.rerun()
                
                processing_time = round(time.time() - start_time, 2)
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                meta_string = f"⏱️ Processed in {processing_time}s | 📅 {timestamp}"
                st.caption(meta_string)
                
                # Append to the ACTIVE session memory
                active_session["chat_history"].append({"user": query_to_run, "ai": clean_summary})
                active_session["display_messages"].append({
                    "role": "assistant", 
                    "content": clean_summary,
                    "has_chart": has_chart,
                    "chart_type": chart_type,
                    "x_col": x_col,
                    "y_col": y_col,
                    "df": df,
                    "follow_ups": follow_ups,
                    "meta": meta_string
                })
                
            except Exception as e:
                error_msg = f"System Error: {str(e)}. Check your inputs."
                st.error(error_msg)
                active_session["display_messages"].append({"role": "assistant", "content": error_msg})