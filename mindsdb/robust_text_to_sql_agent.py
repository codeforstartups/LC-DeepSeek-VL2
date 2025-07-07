# file: robust_text_to_sql_agent.py
import mindsdb_sdk
import sys
import time

# --------------------
# CONFIGURATION
# --------------------
MINDSDB_PARAMS = {}  # e.g., {"host": "...", "port": 47334}
PG = {
    "user": "langchain_user",
    "password": "langchain_password",
    "host": "13.59.72.219",
    "port": "5432",
    "database": "langchain_dev",
}
DB_NAME = "langchain_pg_db"
ENGINE_NAME = "ollama_deepseek_engine"
ENGINE_CONN = {"ollama_serve_url": "http://host.docker.internal:11434"}
MODEL_ALIAS = "text2sql_deepseek"
MODEL_NAME = "deepseek-r1:1.5b"
SKILL_NAME = "text2sql_skill"
AGENT_NAME = "langchain_sql_agent"

def main():
    try:
        server = mindsdb_sdk.connect(**MINDSDB_PARAMS)
        print("✅ Connected to MindsDB")
    except Exception as e:
        print("❌ Cannot connect to MindsDB:", e, file=sys.stderr)
        sys.exit(1)

    # Full clean-up on every run
    for collection, name in [
        ("agents", AGENT_NAME),
        ("skills", SKILL_NAME),
        ("models", MODEL_ALIAS),
        ("databases", DB_NAME)
    ]:
        try:
            getattr(server, collection).delete(name)
            print(f"🗑️ Deleted {collection[:-1]} '{name}'")
        except Exception:
            pass  # not existing is fine

    # Recreate database connection
    server.databases.create(
        name=DB_NAME, engine="postgres", connection_args=PG
    )
    print(f"✅ Created database '{DB_NAME}'")

    # Ensure ML engine exists
    try:
        server.ml_engines.get(ENGINE_NAME)
        print(f"✅ ML engine '{ENGINE_NAME}' already exists")
    except Exception:
        server.ml_engines.create(
            name=ENGINE_NAME,
            handler="ollama",
            connection_data=ENGINE_CONN
        )
        print(f"✅ Created ML engine '{ENGINE_NAME}'")

    # Create LLM model for text-to-SQL conversion
    model = server.models.create(
        name=MODEL_ALIAS,
        engine=ENGINE_NAME,
        predict='completion',
        # Must include input key mapping:
        options={
            'model_name': MODEL_NAME,
            'prompt_template': '{{question}} Please provide the SQL query.',
            'input_column': 'question'
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

    # Create Text-to-SQL skill
    server.skills.create(
        name=SKILL_NAME,
        type="sql",
        params={
            "database": DB_NAME,
            "tables": [],  # Optionally list your tables
            "description": "Text-to-SQL skill over langchain_dev"
        }
    )
    print("✅ Skill created")

    # Create agent linking them
    agent = server.agents.create(
        name=AGENT_NAME,
        model=model,
        skills=[SKILL_NAME]
    )
    print("✅ Agent created")

    # Test the pipeline
    question = "How many users are there in total?"
    reply = agent.completion([{"question": question, "answer": None}])

    sql = getattr(reply, "sql", None)
    answer = getattr(reply, "answer", reply.content)
    print("\n🧪 Generated SQL:\n", sql or "SQL not generated.")
    print("✅ Agent Answer:\n", answer)

if __name__ == "__main__":
    main()
