# file: gemini_text_to_sql_agent.py
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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

DB_NAME = "langchain_pg_db"
ENGINE_NAME = "gemini_engine"
MODEL_ALIAS = "text2sql_gemini"
MODEL_NAME = "gemini-pro"  # Or "gemini-1.5-pro-latest"
SKILL_NAME = "text2sql_skill"
AGENT_NAME = "langchain_sql_agent"

def main():
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        print("🛑 Please set the GEMINI_API_KEY environment variable.", file=sys.stderr)
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
            handler="google_gemini",
            connection_data={"google_api_key": GEMINI_API_KEY}
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
        type="text_to_sql",
        params={
            "database": DB_NAME,
            "tables": ["users"],
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
    question = "How many users signed up in each month during 2023?"

    # First, get the generated SQL from the model directly
    sql_model = server.models.get(MODEL_ALIAS)
    sql_reply = sql_model.predict({'question': question})
    generated_sql = sql_reply['completion'].iloc[0]

    # Then, get the final answer from the agent
    reply = agent.completion([{"question": question, "answer": None}])
    answer = getattr(reply, "answer", reply.content)

    print("\n🧪 Generated SQL:\n", generated_sql or "SQL not generated.")
    print("✅ Agent Answer:\n", answer)


if __name__ == "__main__":
    main()
