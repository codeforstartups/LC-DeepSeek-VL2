import mindsdb_sdk

# Connect to MindsDB
server = mindsdb_sdk.connect('http://127.0.0.1:47334')

# 1️⃣ Create Ollama ML engine
engine = server.ml_engines.create(
    name='ollama_engine',
    handler='ollama',
    connection_data={'ollama_serve_url': 'http://13.59.72.219:11434'}
)

# 2️⃣ Create model using DeepSeek‑R1
model = server.models.create(
    name='tsql_deepseek',
    predict='completion',
    engine=engine.name,
    options={'model_name': 'deepseek-r1:1.5b'}
)
model.wait_for_complete()

# 3️⃣ Connect Postgres DB
db = server.databases.create(
    name='pg_langchain',
    engine='postgres',
    connection_args={
        "dbname":"langchain_dev","user":"langchain_user",
        "password":"langchain_password","host":"13.59.72.219","port":"5432"
    }
)

# 4️⃣ Set up text‑to‑SQL skill
sql_skill = server.skills.create(
    name='sql_skill',
    type='sql',
    params={'database':db.name,'tables':[]}
)

# 5️⃣ Build agent
agent = server.agents.create(
    name='deepseek_sql_agent',
    model=model,
    skills=[sql_skill]
)

# 6️⃣ Ask a question
resp = agent.completion([{"question":"How many users are there in total?","answer":None}])
print(resp.content)
