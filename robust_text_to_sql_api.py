# robust_text_to_sql_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mindsdb_sdk
import sys
import time
import os
import asyncio
import pandas as pd
from sqlalchemy import create_engine
from typing import Optional, Dict, Any, List
import traceback

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed")

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
    MINDSDB_PARAMS = {}
    print("🏠 Connecting to local MindsDB server")

PG = {
    "user": "langchain_user",
    "password": "langchain_password",
    "host": "13.59.72.219",
    "port": "5432",
    "database": "langchain_dev",
}

# Service Configuration
DB_NAME = "langchain_pg_db"
ENGINE_NAME = "ollama_deepseek_engine"
ENGINE_CONN = {"ollama_serve_url": "http://13.59.72.219:11434"}
MODEL_ALIAS = "text2sql_deepseek"
MODEL_NAME = "duckdb-nsql:7b"
SKILL_NAME = "text2sql_skill"
AGENT_NAME = "langchain_sql_agent_deepseek"

# Retry Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# --------------------
# PYDANTIC MODELS
# --------------------

class QueryRequest(BaseModel):
    question: str

class TimingInfo(BaseModel):
    generation_time: float
    execution_time: float
    total_time: float

class QueryResponse(BaseModel):
    success: bool
    sql: Optional[str] = None
    data: Optional[List[Dict[Any, Any]]] = None
    row_count: Optional[int] = None
    timing: Optional[TimingInfo] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    mindsdb_connected: bool
    model_status: Optional[str] = None

# --------------------
# FASTAPI APP
# --------------------

app = FastAPI(
    title="Robust Text-to-SQL API",
    description="Convert natural language questions to SQL queries using MindsDB + DuckDB-NSQL model",
    version="1.0.0"
)

# Global MindsDB server connection
server = None
model_ready = False

# --------------------
# UTILITY FUNCTIONS
# --------------------

def retry_on_failure(max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Decorator for retrying failed operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"❌ Final attempt failed for {func.__name__}: {e}")
                        raise
                    print(f"⚠️ Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    print(f"🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

def check_resource_exists(collection, name):
    """Check if a MindsDB resource (model, skill, agent, database) exists"""
    try:
        resource = getattr(server, collection).get(name)
        return resource is not None
    except Exception:
        return False

def wait_for_model_ready(model, timeout=300):
    """Wait for model to be ready with timeout"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            status = model.get_status()
            print(f"📊 Model status: {status}")

            if status in ("finished", "complete"):
                print("✅ Model is ready!")
                return True
            elif status == "error":
                raise Exception("Model creation failed with error status")

            time.sleep(5)
        except Exception as e:
            print(f"⚠️ Error checking model status: {e}")
            time.sleep(5)

    raise Exception(f"Model not ready after {timeout} seconds")

# --------------------
# SERVICE FUNCTIONS
# --------------------

@retry_on_failure()
def initialize_mindsdb():
    """Initialize MindsDB connection and setup models"""
    global server, model_ready

    try:
        print("🔄 Initializing MindsDB connection...")
        server = mindsdb_sdk.connect(**MINDSDB_PARAMS)
        print("✅ Connected to MindsDB")

        # Setup database, engine, model, skill, agent
        setup_mindsdb_components()
        model_ready = True
        print("✅ MindsDB service ready!")

    except Exception as e:
        print(f"❌ MindsDB initialization failed: {e}")
        model_ready = False
        raise

def setup_mindsdb_components():
    """Setup MindsDB database, model, skill, and agent (only if they don't exist)"""
    global server

    print("🔧 Setting up MindsDB components...")

    # 1. Setup Database
    try:
        if check_resource_exists("databases", DB_NAME):
            print(f"✅ Database '{DB_NAME}' already exists")
        else:
            server.databases.create(
                name=DB_NAME, engine="postgres", connection_args=PG
            )
            print(f"✅ Created database '{DB_NAME}'")
    except Exception as e:
        print(f"⚠️ Database setup issue: {e}")

    # 2. Setup Ollama Engine
    try:
        if check_resource_exists("ml_engines", ENGINE_NAME):
            print(f"✅ Engine '{ENGINE_NAME}' already exists")
        else:
            server.ml_engines.create(
                name=ENGINE_NAME,
                handler="ollama",
                connection_data=ENGINE_CONN
            )
            print(f"✅ Created engine '{ENGINE_NAME}'")
    except Exception as e:
        print(f"⚠️ Engine setup issue: {e}")

    # 3. Setup Model
    try:
        if check_resource_exists("models", MODEL_ALIAS):
            print(f"✅ Model '{MODEL_ALIAS}' already exists")
            model = server.models.get(MODEL_ALIAS)

            # Check if model is ready
            status = model.get_status()
            if status not in ("finished", "complete"):
                print(f"📊 Model exists but not ready (status: {status}), waiting...")
                wait_for_model_ready(model)
        else:
            print(f"🔄 Creating model '{MODEL_ALIAS}'...")
            model = server.models.create(
                name=MODEL_ALIAS,
                predict='completion',
                engine=ENGINE_NAME,
                options={'model_name': MODEL_NAME}
            )
            print("⏳ Model creation started...")
            wait_for_model_ready(model)

    except Exception as e:
        print(f"❌ Model setup failed: {e}")
        raise

    # 4. Setup Skill
    try:
        if check_resource_exists("skills", SKILL_NAME):
            print(f"✅ Skill '{SKILL_NAME}' already exists")
        else:
            server.skills.create(
                name=SKILL_NAME,
                type="sql",
                params={
                    "database": DB_NAME,
                    "tables": ["users"],
                    "description": "Users table: id (int), username (text), email (text), signup_date (date), status (text)"
                }
            )
            print(f"✅ Created skill '{SKILL_NAME}'")
    except Exception as e:
        print(f"⚠️ Skill setup issue: {e}")

    # 5. Setup Agent
    try:
        if check_resource_exists("agents", AGENT_NAME):
            print(f"✅ Agent '{AGENT_NAME}' already exists")
        else:
            model = server.models.get(MODEL_ALIAS)
            server.agents.create(
                name=AGENT_NAME,
                model=model,
                skills=[SKILL_NAME],
                params={
                    'prompt_template': 'Schema: users table has id, username, email, signup_date, status columns. Use id not user_id.'
                }
            )
            print(f"✅ Created agent '{AGENT_NAME}'")
    except Exception as e:
        print(f"⚠️ Agent setup issue: {e}")

@retry_on_failure()
async def execute_text_to_sql_query(question: str) -> QueryResponse:
    """Execute text-to-SQL query and return results with retry logic"""
    if not model_ready or server is None:
        raise HTTPException(status_code=503, detail="MindsDB service not ready")

    try:
        print(f"🔍 Processing question: {question}")

        # Start timing SQL generation
        start_time = time.time()

        # Sanitize input to prevent SQL injection
        sanitized_question = question.replace("'", "''")

        # Query model directly
        result = server.query(f"""
            SELECT text, completion
            FROM {MODEL_ALIAS}
            WHERE text = '{sanitized_question}'
        """)

        df = result.fetch()
        generation_time = time.time() - start_time

        if df.empty:
            return QueryResponse(
                success=False,
                error="No results returned from model"
            )

        sql_query = df.iloc[0]['completion'].strip()
        print(f"✅ Generated SQL (took {generation_time:.2f}s):\n{sql_query}")

        # Start timing SQL execution
        exec_start_time = time.time()

        # Execute on PostgreSQL database
        connection_string = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['database']}"
        engine = create_engine(connection_string)
        data = pd.read_sql_query(sql_query, engine)
        engine.dispose()

        execution_time = time.time() - exec_start_time
        total_time = generation_time + execution_time

        # Convert DataFrame to list of dictionaries
        data_dict = data.to_dict('records')

        print(f"📊 Query executed successfully: {len(data_dict)} rows")

        return QueryResponse(
            success=True,
            sql=sql_query,
            data=data_dict,
            row_count=len(data_dict),
            timing=TimingInfo(
                generation_time=generation_time,
                execution_time=execution_time,
                total_time=total_time
            )
        )

    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        print(f"❌ {error_msg}")
        return QueryResponse(
            success=False,
            error=error_msg
        )

# --------------------
# API ENDPOINTS
# --------------------

@app.on_event("startup")
async def startup_event():
    """Initialize MindsDB on startup"""
    try:
        initialize_mindsdb()
    except Exception as e:
        print(f"❌ Startup failed: {e}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Robust Text-to-SQL API",
        "version": "1.0.0",
        "status": "active" if model_ready else "initializing",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /health",
            "examples": "GET /examples"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check MindsDB connection
        mindsdb_connected = server is not None and model_ready

        model_status = None
        if mindsdb_connected:
            try:
                model = server.models.get(MODEL_ALIAS)
                model_status = model.get_status()
            except:
                model_status = "unknown"

        return HealthResponse(
            status="healthy" if mindsdb_connected else "unhealthy",
            service="robust-text-to-sql-api",
            mindsdb_connected=mindsdb_connected,
            model_status=model_status
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            service="robust-text-to-sql-api",
            mindsdb_connected=False,
            model_status=f"error: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main text-to-SQL query endpoint with retry logic"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return await execute_text_to_sql_query(request.question)

@app.get("/examples")
async def get_examples():
    """Get example queries for testing"""
    return {
        "examples": [
            "How many users signed up in each month during 2023?",
            "What is the total number of users in the database?",
            "Show me the latest 5 users who signed up",
            "Find users who signed up in July 2023",
            "Show users whose username starts with 'a'",
            "Count how many users signed up each day in August 2023",
            "Find the first and last user to sign up",
            "List all unique signup months in the database"
        ]
    }

@app.get("/status")
async def get_status():
    """Get detailed service status including resource existence"""
    try:
        if server is None:
            return {"error": "MindsDB not connected"}

        status = {
            "service": "robust-text-to-sql-api",
            "mindsdb_connected": True,
            "resources": {}
        }

        # Check each resource
        resources = [
            ("database", "databases", DB_NAME),
            ("engine", "ml_engines", ENGINE_NAME),
            ("model", "models", MODEL_ALIAS),
            ("skill", "skills", SKILL_NAME),
            ("agent", "agents", AGENT_NAME)
        ]

        for resource_type, collection, name in resources:
            exists = check_resource_exists(collection, name)
            status["resources"][f"{resource_type}_{name}"] = {
                "exists": exists,
                "type": resource_type
            }

            # Get model status if it's a model
            if resource_type == "model" and exists:
                try:
                    model = server.models.get(name)
                    model_status = model.get_status()
                    status["resources"][f"{resource_type}_{name}"]["status"] = model_status
                except:
                    status["resources"][f"{resource_type}_{name}"]["status"] = "unknown"

        return status

    except Exception as e:
        return {
            "service": "robust-text-to-sql-api",
            "mindsdb_connected": False,
            "error": str(e)
        }

# --------------------
# MAIN
# --------------------

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Robust Text-to-SQL API on port 8009...")
    uvicorn.run(app, host="0.0.0.0", port=8009)
