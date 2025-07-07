# file: openai_text_to_sql_agent.py
import mindsdb_sdk
import os
import sys
import time

# --------------------
# CONFIGURATION
# --------------------
MINDSDB_PARAMS = {}
PG = {"user": "langchain_user", "password": "langchain_password",
      "host": "13.59.72.219", "port": "5432", "database": "langchain_dev"}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

DB_NAME = "langchain_pg_db"
ENGINE_NAME = "openai_engine"
AGENT_NAME = "text2sql_agent"

def main():
    if not OPENAI_API_KEY:
        print("🛑 Please set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    server = mindsdb_sdk.connect(**MINDSDB_PARAMS)
    print("✅ Connected to MindsDB")

    # -- Cleanup resources
    for coll, name in [("agents", AGENT_NAME)]:
        try: getattr(server, coll).drop(name); print(f"Dropped {name}")
        except: pass
    for coll, name in [("skills", "text2sql_skill")]:
        try: getattr(server, coll).drop(name); print(f"Dropped {name}")
        except: pass
    for coll, name in [("databases", DB_NAME)]:
        try: getattr(server, coll).drop(name); print(f"Dropped {name}")
        except: pass

    # -- Create database connection
    server.databases.create(DB_NAME, engine="postgres", connection_args=PG)
    print("✅ Database connected:", DB_NAME)

    # -- Create ML engine
    try:
        server.ml_engines.get(ENGINE_NAME)
    except:
        server.ml_engines.create(
            name=ENGINE_NAME, handler="openai",
            connection_data={"openai_api_key": OPENAI_API_KEY}
        )
    print("✅ Engine ready:", ENGINE_NAME)

    # -- Create SQL skill
    sql_skill = server.skills.create(
        name="text2sql_skill",
        type="sql",
        params={"database": DB_NAME, "tables": ["users"],
                "description": "Text-to-SQL over users table"}
    )
    print("✅ Skill created:", sql_skill.name)

    # -- Create the agent with built-in prompting
    agent = server.agents.create(name=AGENT_NAME, skills=[sql_skill])
    print("✅ Agent created:", agent.name)

    # -- Ask and execute SQL
    question = "How many users signed up in each month during 2023?"
    reply = agent.completion([{"question": question, "answer": None}])

    print("\n🧪 Generated SQL:\n", reply.sql)
    print("✅ Query Results:\n", reply.answer)

if __name__ == "__main__":
    main()
