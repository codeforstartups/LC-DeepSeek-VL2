# file: robust_text_to_sql_agent.py
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

DB_NAME = "langchain_pg_db"
ENGINE_NAME = "ollama_deepseek_engine"
ENGINE_CONN = {"ollama_serve_url": "http://13.59.72.219:11434"}
MODEL_ALIAS = "text2sql_deepseek"
MODEL_NAME = "duckdb-nsql:7b"
SKILL_NAME = "text2sql_skill"
AGENT_NAME = "langchain_sql_agent_deepseek"

def execute_text_to_sql(server, question):
    """Execute text-to-SQL query using the robust SQL-based approach"""
    try:
        print(f"\n🔍 Processing question: {question}")

        # Start timing SQL generation
        start_time = time.time()

        # Query model directly using standard Ollama pattern
        result = server.query(f"""
            SELECT text, completion
            FROM {MODEL_ALIAS}
            WHERE text = '{question}'
        """)

        df = result.fetch()

        # Calculate SQL generation time
        generation_time = time.time() - start_time

        if df.empty:
            print("❌ No results returned from agent")
            return None, None, 0

        sql_query = df.iloc[0]['completion']
        print(f"✅ Generated SQL (took {generation_time:.2f}s):\n{sql_query}")

        # Start timing SQL execution
        exec_start_time = time.time()

        # Execute on PostgreSQL database using SQLAlchemy
        connection_string = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['database']}"
        engine = create_engine(connection_string)
        data = pd.read_sql_query(sql_query, engine)
        engine.dispose()

        # Calculate SQL execution time
        execution_time = time.time() - exec_start_time
        total_time = generation_time + execution_time

        print(f"\n📊 Query Results (executed in {execution_time:.3f}s):")
        print(data)
        print(f"⏱️ Total time: {total_time:.2f}s (Generation: {generation_time:.2f}s, Execution: {execution_time:.3f}s)")

        return sql_query, data, total_time

    except Exception as e:
        print(f"❌ Error executing text-to-SQL: {e}")
        return None, None, 0

def main():
    try:
        server = mindsdb_sdk.connect(**MINDSDB_PARAMS)
        print("✅ Connected to MindsDB")
    except Exception as e:
        print("❌ Cannot connect to MindsDB:", e, file=sys.stderr)
        sys.exit(1)

    # 🔁 Full Cleanup
    for collection, name in [
        ("agents", AGENT_NAME),
        ("skills", SKILL_NAME)
    ]:
        try:
            getattr(server, collection).drop(name)
            print(f"🗑️ Dropped {collection[:-1]} '{name}'")
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

    # ✅ Ensure Ollama engine
    try:
        server.ml_engines.get(ENGINE_NAME)
        print(f"✅ Engine '{ENGINE_NAME}' already exists")
    except:
        server.ml_engines.create(
            name=ENGINE_NAME,
            handler="ollama",
            connection_data=ENGINE_CONN
        )
        print(f"✅ Created engine '{ENGINE_NAME}'")

    # 🧠 Create DeepSeek model with enhanced prompt
    model = server.models.create(
        name=MODEL_ALIAS,
        predict='completion',
        engine=ENGINE_NAME,
        options={
            'model_name': MODEL_NAME
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
            "description": "Users table: id (int), username (text), email (text), signup_date (date), status (text)"
        }
    )
    print("✅ Skill created")

    # 🤖 Create agent
    agent = server.agents.create(
        name=AGENT_NAME,
        model=model,
        skills=[SKILL_NAME],
        params={
            'prompt_template': 'Schema: users table has id, username, email, signup_date, status columns. Use id not user_id.'
        }
    )
    print("✅ Agent created")

    # 🧪 Test using the robust SQL-based approach with diverse queries
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
    print("🚀 TESTING DUCKDB-NSQL TEXT-TO-SQL AGENT")
    print("="*50)

    # Performance tracking
    successful_queries = 0
    failed_queries = 0
    total_generation_time = 0
    total_execution_time = 0
    all_times = []

    for i, question in enumerate(questions, 1):
        print(f"\n📝 Test {i}/{len(questions)}")
        print("-" * 50)

        sql, data, total_time = execute_text_to_sql(server, question)

        if sql and data is not None:
            successful_queries += 1
            all_times.append(total_time)
            print(f"✅ Success! Generated {len(data)} rows")
        else:
            failed_queries += 1
            print("❌ Failed to generate or execute SQL")

        if i < len(questions):
            print("\n" + "⏳ Waiting 2 seconds before next test...")
            time.sleep(2)

    print("\n" + "="*60)
    print("🎉 DUCKDB-NSQL TEXT-TO-SQL TESTING COMPLETE!")
    print("="*60)

    # Performance Summary
    print("\n📊 PERFORMANCE SUMMARY:")
    print("-" * 30)
    print(f"✅ Successful Queries: {successful_queries}/{len(questions)} ({successful_queries/len(questions)*100:.1f}%)")
    print(f"❌ Failed Queries: {failed_queries}")

    if all_times:
        avg_time = sum(all_times) / len(all_times)
        min_time = min(all_times)
        max_time = max(all_times)
        print(f"⏱️ Average Response Time: {avg_time:.2f}s")
        print(f"⚡ Fastest Query: {min_time:.2f}s")
        print(f"🐌 Slowest Query: {max_time:.2f}s")

    print("="*60)

if __name__ == "__main__":
    main()
