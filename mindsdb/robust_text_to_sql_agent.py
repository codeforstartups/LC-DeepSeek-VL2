# file: robust_text_to_sql_agent.py
import mindsdb_sdk
import sys
import time

# --------------------
# CONFIGURATION
# --------------------
MINDSDB_PARAMS = {}
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
    # 1️⃣ Connect to MindsDB
    try:
        server = mindsdb_sdk.connect(**MINDSDB_PARAMS)
        print("✅ Connected to MindsDB")
    except Exception as e:
        print("❌ Cannot connect to MindsDB:", e, file=sys.stderr)
        sys.exit(1)

    # 2️⃣ Ensure Postgres integration
    try:
        server.databases.get(DB_NAME)
        print(f"✅ Database '{DB_NAME}' already exists")
    except Exception:
        server.databases.create(name=DB_NAME, engine="postgres", connection_args=PG)
        print(f"✅ Created database '{DB_NAME}'")

    # 3️⃣ Register Ollama engine
    try:
        server.ml_engines.get(ENGINE_NAME)
        print(f"✅ ML engine '{ENGINE_NAME}' already exists")
    except Exception:
        server.ml_engines.create(name=ENGINE_NAME, handler="ollama", connection_data=ENGINE_CONN)
        print(f"✅ Created ML engine '{ENGINE_NAME}'")

    # 4️⃣ Create conversational model
    model = server.models.create(
        name=MODEL_ALIAS,
        predict="completion",
        engine=ENGINE_NAME,
        options={"model_name": MODEL_NAME}
    )
    print("⏳ Model creation initiated...")

    # ⏱️ Poll for status until complete/error
    status = model.get_status()
    print(f"Model status: {status}")
    while status not in ("finished", "error", "complete"):
        time.sleep(5)
        status = model.get_status()
        print(f"Model status: {status}")
    if status in ("error",):
        print("❌ Model training failed. Exiting.")
        sys.exit(1)
    print(f"✅ Model '{MODEL_ALIAS}' is ready")

    # 5️⃣ Create Text-to-SQL skill
    try:
        server.skills.get(SKILL_NAME)
        print(f"✅ Skill '{SKILL_NAME}' already exists")
    except Exception:
        server.skills.create(name=SKILL_NAME, type="sql", params={"database": DB_NAME, "tables": []})
        print(f"✅ Created skill '{SKILL_NAME}'")

    # 6️⃣ Instantiate agent
    try:
        agent = server.agents.get(AGENT_NAME)
        print(f"✅ Agent '{AGENT_NAME}' already exists")
    except Exception:
        agent = server.agents.create(name=AGENT_NAME, model=model, skills=[SKILL_NAME])
        print(f"✅ Created agent '{AGENT_NAME}'")

    # 7️⃣ Ask a question
    question = "How many users are there in total?"
    reply = agent.completion([{"question": question, "answer": None}])
    print("\n🧪 Generated SQL:\n", getattr(reply, "sql", "N/A"))
    print("✅ Agent Answer:\n", getattr(reply, "answer", reply.content))

if __name__ == "__main__":
    main()
