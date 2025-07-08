# file: openai_text_to_sql_agent.py
import mindsdb_sdk
import sys
import time
import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads the .env file
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("💡 Or set environment variables manually")

# --------------------
# CONFIGURATION
# --------------------

# Environment Configuration
USE_REMOTE_MINDSDB = os.getenv('USE_REMOTE_MINDSDB', 'true').lower() == 'true'
REMOTE_MINDSDB_HOST = os.getenv('REMOTE_MINDSDB_HOST', "13.59.72.219")
REMOTE_MINDSDB_PORT = int(os.getenv('REMOTE_MINDSDB_PORT', 47334))

# MindsDB Connection Parameters
if USE_REMOTE_MINDSDB:
    MINDSDB_PARAMS = {
        'url': f'http://{REMOTE_MINDSDB_HOST}:{REMOTE_MINDSDB_PORT}'
    }
    print(f"🌐 Connecting to remote MindsDB server: {REMOTE_MINDSDB_HOST}:{REMOTE_MINDSDB_PORT}")
else:
    MINDSDB_PARAMS = {}  # Local MindsDB (default port 47334)
    print("🏠 Connecting to local MindsDB server")

PG = {
    "user": "langchain_user",
    "password": "langchain_password",
    "host": "13.59.72.219",
    "port": "5432",
    "database": "langchain_dev",
}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

DB_NAME = "langchain_pg_db"
ENGINE_NAME = "openai_engine"
MODEL_ALIAS = "text2sql_openai"
MODEL_NAME = "gpt-4" # "gpt-4o-mini"
SKILL_NAME = "text2sql_skill"
AGENT_NAME = "langchain_sql_agent"

def execute_text_to_sql(server, question):
    """Execute text-to-SQL query using the robust SQL-based approach"""
    try:
        print(f"\n🔍 Processing question: {question}")

        # Query agent directly using SQL (ROBUST APPROACH)
        result = server.query(f"""
            SELECT answer
            FROM {AGENT_NAME}
            WHERE question = '{question}'
        """)

        df = result.fetch()
        if df.empty:
            print("❌ No results returned from agent")
            return None, None

        sql_query = df.iloc[0]['answer']
        print(f"✅ Generated SQL:\n{sql_query}")

        # Execute on PostgreSQL database using SQLAlchemy
        connection_string = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['database']}"
        engine = create_engine(connection_string)
        data = pd.read_sql_query(sql_query, engine)
        engine.dispose()

        print(f"\n📊 Query Results:")
        print(data)

        return sql_query, data

    except Exception as e:
        print(f"❌ Error executing text-to-SQL: {e}")
        return None, None

def main():
    if not OPENAI_API_KEY:
        print("🛑 Please set the OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    try:
        server = mindsdb_sdk.connect(**MINDSDB_PARAMS)
        print("✅ Connected to MindsDB")
    except Exception as e:
        print("❌ Cannot connect to MindsDB:", e, file=sys.stderr)
        sys.exit(1)

    # 🔁 Full Cleanup
    for coll, name in [("agents", AGENT_NAME), ("skills", SKILL_NAME)]:
        try:
            getattr(server, coll).drop(name)
            print(f"🗑️ Dropped {coll[:-1]} '{name}'")
        except:
            pass

    for model_or_db, name in [("models", MODEL_ALIAS), ("databases", DB_NAME)]:
        try:
            getattr(server, model_or_db).drop(name)
            print(f"🗑️ Dropped {model_or_db[:-1]} '{name}'")
        except:
            pass

    # 🔧 Recreate database
    server.databases.create(
        name=DB_NAME, engine="postgres", connection_args=PG
    )
    print(f"✅ Created database '{DB_NAME}'")

    # ✅ Ensure OpenAI engine exists
    try:
        server.ml_engines.get(ENGINE_NAME)
        print(f"✅ Engine '{ENGINE_NAME}' already exists")
    except:
        server.ml_engines.create(
            name=ENGINE_NAME,
            handler="openai",
            connection_data={"openai_api_key": OPENAI_API_KEY}
        )
        print(f"✅ Created engine '{ENGINE_NAME}'")

    # 🧠 Create LLM model
    model = server.models.create(
        name=MODEL_ALIAS,
        engine=ENGINE_NAME,
        predict='completion',
        options={
            'model_name': MODEL_NAME,
            'prompt_template': '''
You are a PostgreSQL expert. Generate ONLY valid SQL queries based on the user's question.

Rules:
1. Return ONLY the SQL query, no explanation
2. No markdown formatting
3. Use proper PostgreSQL syntax
4. Quote table/column names with backticks if needed

Database Schema:
- users table: id, username, email, signup_date, status

User question: {{question}}
'''.strip()
        }
    )
    print("⏳ Model creation started...")
    status = model.get_status()
    while status not in ("finished", "complete", "error"):
        time.sleep(5)
        status = model.get_status()
        print("Model status:", status)
    if status == "error":
        print("❌ Model creation failed")
        sys.exit(1)
    print("✅ Model is ready")

    # 🛠️ Create SQL skill
    server.skills.create(
        name=SKILL_NAME,
        type="sql",
        params={
            "database": DB_NAME,
            "tables": ["users"],  # Adjust to your schema
            "description": "Text-to-SQL skill over 'users' table"
        }
    )
    print("✅ Skill created")

    # 🤖 Create agent
    agent = server.agents.create(
        name=AGENT_NAME,
        model=model,
        skills=[SKILL_NAME]
    )
    print("✅ Agent created")

    # 🧪 Test using the robust SQL-based approach
    questions = [
        "How many users signed up in each month during 2023?",
        "What is the total number of users in the database?",
        "Show me the latest 5 users who signed up",
        "Find users who signed up in July 2023",
        "Get the average signup date by month",
        "Show users whose username starts with 'a'",
        "Count how many users signed up each day in August 2023",
        "Find the first and last user to sign up",
        "List all unique signup months in the database",
        "Show users who signed up on weekends"
    ]

    print("\n" + "="*50)
    print("🚀 TESTING TEXT-TO-SQL AGENT")
    print("="*50)

    for i, question in enumerate(questions, 1):
        print(f"\n📝 Test {i}/{len(questions)}")
        print("-" * 30)

        sql, data = execute_text_to_sql(server, question)

        if sql and data is not None:
            print(f"✅ Success! Generated {len(data)} rows")
        else:
            print("❌ Failed to generate or execute SQL")

        if i < len(questions):
            print("\n" + "⏳ Waiting 2 seconds before next test...")
            time.sleep(2)

    print("\n" + "="*50)
    print("🎉 TEXT-TO-SQL TESTING COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()
