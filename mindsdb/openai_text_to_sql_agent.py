# file: openai_text_to_sql_agent.py
import mindsdb_sdk, os, sys, time

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = "langchain_pg_db"
ENGINE_NAME = "openai_engine"
SKILL_NAME = "text2sql_skill"
AGENT_NAME = "text2sql_agent"

def main():
    if not OPENAI_API_KEY:
        print("🛑 Please set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    server = mindsdb_sdk.connect(**MINDSDB_PARAMS)
    print("✅ Connected to MindsDB")

    # Full cleanup
    for coll, name in [("agents", AGENT_NAME), ("skills", SKILL_NAME),
                       ("models", AGENT_NAME), ("databases", DB_NAME)]:
        try:
            getattr(server, coll).drop(name)
            print(f"🗑️ Dropped {name}")
        except:
            pass

    # Re-create database
    server.databases.create(DB_NAME, engine="postgres", connection_args=PG)
    print("✅ Created database")

    # Ensure engine exists
    try:
        server.ml_engines.get(ENGINE_NAME)
    except:
        server.ml_engines.create(
            name=ENGINE_NAME,
            handler="openai",
            connection_data={"openai_api_key": OPENAI_API_KEY}
        )
    print("✅ Engine ready")

    # Create SQL skill
    sql_skill = server.skills.create(
        name=SKILL_NAME,
        type="sql",
        params={
            "database": DB_NAME,
            "tables": ["users"],
            "description": "Text‑to‑SQL over 'users'"
        }
    )
    print("✅ Skill created:", sql_skill.name)

    # Create agent using the **skill name** (not skill object)
    agent = server.agents.create(
        name=AGENT_NAME,
        skills=[sql_skill.name]
    )
    print("✅ Agent created:", agent.name)

    # Execute a query
    question = "How many users signed up in each month during 2023?"
    reply = agent.completion([{"question": question, "answer": None}])

    print("\n🧪 SQL:", reply.sql)
    print("✅ Results:\n", reply.answer)

if __name__ == "__main__":
    main()
