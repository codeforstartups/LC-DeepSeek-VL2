# file: openai_text_to_sql_agent.py
import mindsdb_sdk
import sys
import time
import os

# --------------------
# CONFIGURATION
# --------------------
MINDSDB_PARAMS = {}  # e.g., connect to local or remote instance
PG = {
    "user": "langchain_user",
    "password": "langchain_password",
    "host": "13.59.72.219",
    "port": "5432",
    "database": "langchain_dev",
}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

DB_NAME = "langchain_pg_db"
ENGINE_NAME = "openai_engine"
MODEL_ALIAS = "text2sql_openai"
MODEL_NAME = "gpt-4o-mini"  # Or "gpt-4"
SKILL_NAME = "text2sql_skill"
AGENT_NAME = "langchain_sql_agent"

def main():
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("🛑 Please set the OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    try:
        server = mindsdb_sdk.connect(**MINDSDB_PARAMS)
        print("✅ Connected to MindsDB")
    except Exception as e:
        print("❌ Cannot connect to MindsDB:", e, file=sys.stderr)
        sys.exit(1)

    # 🔁 Full Cleanup
    for collection, name in [
        ("agents", AGENT_NAME),
        ("skills", SKILL_NAME),
    ]:
        try:
            getattr(server, collection).drop(name)
            print(f"🗑️ Dropped {collection[:-1]} '{name}'")
        except:
            pass

    try:
        server.models.drop(MODEL_ALIAS)
        print(f"🗑️ Dropped model '{MODEL_ALIAS}'")
    except:
        pass

    try:
        server.databases.drop(DB_NAME)
        print(f"🗑️ Dropped database '{DB_NAME}'")
    except:
        pass

    # 🔧 Recreate database
    server.databases.create(
        name=DB_NAME, engine="postgres", connection_args=PG
    )
    print(f"✅ Created database '{DB_NAME}'")

    # ✅ Ensure ML engine
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

    # 🧠 Create model with input mapping
    model = server.models.create(
        name=MODEL_ALIAS,
        predict='completion',
        engine=ENGINE_NAME,
        options={
            'model_name': MODEL_NAME,
            'prompt_template': '''
You are a world-class SQL generation-specific model.
Your sole purpose is to generate a valid SQL query given a user's question and the database schema.
Do not provide any explanation or natural language text, only the SQL query.

Here is the database schema:
{{database_schema}}

Here is the user's question:
{{question}}
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
        print("❌ Model creation failed"); sys.exit(1)
    print("✅ Model is ready")

    # 🛠️ Create SQL skill
    server.skills.create(
        name=SKILL_NAME,
        type="sql",
        params={
            "database": DB_NAME,
            "tables": [],
            "description": "Text-to-SQL skill over langchain_dev"
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

    # 🧪 Test it
    question = "How many users are there in total?"
    reply = agent.completion([{"question": question, "answer": None}])
    sql = getattr(reply, "sql", None)
    answer = getattr(reply, "answer", reply.content)

    print("\n🧪 Generated SQL:\n", sql or "SQL not generated.")
    print("✅ Agent Answer:\n", answer)

if __name__ == "__main__":
    main()
