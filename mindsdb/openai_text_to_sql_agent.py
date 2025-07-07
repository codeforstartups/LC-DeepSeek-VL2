# file: openai_text_to_sql_agent.py
import mindsdb_sdk
import sys
import time
import os

# --------------------
# CONFIGURATION
# --------------------
MINDSDB_PARAMS = {}  # adjust if connecting to remote MindsDB instance
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
MODEL_NAME = "gpt-4o-mini"  # or "gpt-4", "gpt-3.5-turbo", etc.
SKILL_NAME = "text2sql_skill"
AGENT_NAME = "langchain_sql_agent"

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

    # 🔁 Full cleanup
    for coll, name in [("agents", AGENT_NAME), ("skills", SKILL_NAME)]:
        try:
            getattr(server, coll).drop(name)
            print(f"🗑️ Dropped {coll[:-1]} '{name}'")
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

    # ⚙️ Recreate database
    server.databases.create(
        name=DB_NAME, engine="postgres", connection_args=PG
    )
    print(f"✅ Created database '{DB_NAME}'")

    # 🔌 Configure OpenAI engine
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
        # This matches official documentation :contentReference[oaicite:1]{index=1}

    # 🧠 Create LLM model
    model = server.models.create(
        name=MODEL_ALIAS,
        engine=ENGINE_NAME,
        predict='completion',
        options={
            'model_name': MODEL_NAME,
            'prompt_template': '''
You are a world-class SQL generator. Generate only SQL queries—no explanations.

Schema:
{{database_schema}}

User question:
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
        print("❌ Model creation failed")
        sys.exit(1)
    print("✅ Model is ready")

    # 🛠️ Create SQL skill with table reference
    server.skills.create(
        name=SKILL_NAME,
        type="sql",
        params={
            "database": DB_NAME,
            "tables": ["users"],  # specify your relevant tables
            "description": "SQL interface over users table"
        }
    )
    print("✅ Skill created")

    # 🤖 Create the agent
    agent = server.agents.create(
        name=AGENT_NAME,
        model=model,
        skills=[SKILL_NAME]
    )
    print("✅ Agent created")

    # 🧪 Execute a test query
    question = "How many users are there in total?"
    reply = agent.completion([{"question": question, "answer": None}])
    sql = getattr(reply, "sql", None)
    answer = getattr(reply, "answer", reply.content)

    print("\n🧪 Generated SQL:\n", sql or "SQL not generated.")
    print("✅ Agent Answer:\n", answer)

if __name__ == "__main__":
    main()
