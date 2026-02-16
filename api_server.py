from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import duckdb
from datetime import datetime
from quantitative_betting_workflow import QuantitativeBettingWorkflow
import sys
from io import StringIO
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start database sync worker in background
try:
    from database_sync import ensure_database_exists, start_periodic_upload
    
    # Download database on startup
    if ensure_database_exists():
        logger.info("✅ Database ready")
        
        # Start periodic upload to GCS (every hour)
        upload_thread = start_periodic_upload(daemon=True)
        logger.info("🔄 Periodic GCS upload enabled (hourly)")
    else:
        logger.warning("⚠️  Database setup failed, proceeding anyway")
except Exception as e:
    logger.error(f"❌ Database sync initialization failed: {e}")

app = FastAPI(
    title="Quantitative Betting API",
    description="API for automated betting workflow with Cloud Scheduler integration",
    version="1.0.0"
)

# Global state for tracking workflow runs
workflow_status = {
    "last_run": None,
    "last_result": None,
    "is_running": False
}

class WorkflowConfig(BaseModel):
    min_ev_threshold: float = 0.08
    max_ev_threshold: float = 0.70
    min_confidence: float = 0.65
    min_stake: float = 0.10
    max_daily_stake: float = 5.0
    max_bets_per_match: int = 1
    auto_place_bets: bool = True
    target_markets: List[str] = ['btts', 'over_under_15', 'handicap']

class WorkflowResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime
    opportunities_found: Optional[int] = None
    bets_placed: Optional[int] = None
    log: Optional[str] = None

class BetHistoryResponse(BaseModel):
    runner_id: int
    match_name: str
    league: str
    market: str
    runner_name: str
    stake: float
    odds: float
    expected_value: float
    confidence: float
    placed_at: datetime

def run_workflow_task(config: WorkflowConfig):
    """Background task to run the betting workflow"""
    global workflow_status
    
    workflow_status["is_running"] = True
    output_capture = StringIO()
    sys.stdout = output_capture
    
    try:
        logger.info("Starting betting workflow...")
        
        # Build config dictionary for workflow
        workflow_config = {
            'min_ev': config.min_ev_threshold,
            'max_ev': config.max_ev_threshold,
            'min_confidence': config.min_confidence,
            'min_stake': config.min_stake,
            'max_daily_stake': config.max_daily_stake,
            'max_bets_per_match': config.max_bets_per_match,
            'auto_place_bets': config.auto_place_bets,
            'target_markets': config.target_markets
        }
        
        # Initialize workflow
        workflow = QuantitativeBettingWorkflow(config=workflow_config)
        
        # Run workflow
        workflow.run_full_workflow()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        workflow_status["last_run"] = datetime.now()
        workflow_status["last_result"] = {
            "status": "success",
            "log": output_capture.getvalue()
        }
        
        logger.info("Workflow completed successfully")
        
    except Exception as e:
        sys.stdout = sys.__stdout__
        logger.error(f"Workflow failed: {str(e)}")
        
        workflow_status["last_result"] = {
            "status": "error",
            "error": str(e),
            "log": output_capture.getvalue()
        }
    
    finally:
        workflow_status["is_running"] = False

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Quantitative Betting API",
        "version": "1.0.0",
        "endpoints": {
            "POST /run-workflow": "Trigger betting workflow",
            "GET /bet-history": "Get bet history",
            "GET /status": "Get system status",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    try:
        # Check database connection
        conn = duckdb.connect('football_data.duckdb')
        conn.execute("SELECT 1").fetchone()
        conn.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "database": "connected"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/run-workflow", response_model=WorkflowResponse)
async def run_workflow(
    background_tasks: BackgroundTasks,
    config: Optional[WorkflowConfig] = None
):
    """
    Trigger the betting workflow
    
    This endpoint is designed to be called by Cloud Scheduler.
    It runs the workflow in the background and returns immediately.
    """
    if workflow_status["is_running"]:
        return WorkflowResponse(
            status="already_running",
            message="Workflow is already running",
            timestamp=datetime.now()
        )
    
    # Use default config if none provided
    if config is None:
        config = WorkflowConfig()
    
    # Start workflow in background
    background_tasks.add_task(run_workflow_task, config)
    
    logger.info(f"Workflow triggered with config: {config.dict()}")
    
    return WorkflowResponse(
        status="started",
        message="Workflow started successfully",
        timestamp=datetime.now()
    )

@app.get("/workflow-status")
async def get_workflow_status():
    """Get the status of the last workflow run"""
    return {
        "is_running": workflow_status["is_running"],
        "last_run": workflow_status["last_run"],
        "last_result": workflow_status["last_result"]
    }

@app.get("/bet-history", response_model=List[BetHistoryResponse])
async def get_bet_history(
    days: int = 7,
    limit: int = 100
):
    """
    Get bet history
    
    - **days**: Number of days to look back (default: 7)
    - **limit**: Maximum number of bets to return (default: 100)
    """
    try:
        conn = duckdb.connect('football_data.duckdb')
        
        bets = conn.execute(f"""
            SELECT 
                runner_id,
                match_name,
                league,
                market,
                runner_name,
                stake,
                odds,
                expected_value,
                confidence,
                placed_at
            FROM bet_history
            WHERE placed_at >= CURRENT_TIMESTAMP - INTERVAL '{days} days'
            ORDER BY placed_at DESC
            LIMIT {limit}
        """).fetchall()
        
        conn.close()
        
        return [
            BetHistoryResponse(
                runner_id=bet[0],
                match_name=bet[1],
                league=bet[2],
                market=bet[3],
                runner_name=bet[4],
                stake=bet[5],
                odds=bet[6],
                expected_value=bet[7],
                confidence=bet[8],
                placed_at=bet[9]
            )
            for bet in bets
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/current-bets")
async def get_current_bets():
    """
    Get current active bets from Matchbook
    
    Returns live bet status, expected value, and P/L from account
    """
    try:
        from src.matchbook import matchbookExchange
        
        # Connect to Matchbook
        matchbook = matchbookExchange()
        matchbook.login()
        
        # Get account info for overall P/L
        account_data = matchbook.get_account()
        account_balance = account_data.get('balance', 0)
        account_exposure = account_data.get('exposure', 0)
        account_commission_reserve = account_data.get('commission-reserve', 0)
        account_free_funds = account_data.get('free-funds', 0)
        
        # Get current bets
        current_bets_response = matchbook.get_current_bets()
        
        # Get settled bets from past day for P/L calculation
        settled_bets_response = matchbook.get_settled_bets(days=1)
        # Calculate P/L from settled bets
        total_pl = settled_bets_response.get('net-profit-and-loss', 0)
        
        if not current_bets_response:
            return {
                "total_bets": 0,
                "total_stake": 0.0,
                "potential_profit": 0.0,
                "current_pl": total_pl,
                "account": {
                    "balance": account_balance,
                    "exposure": account_exposure,
                    "commission_reserve": account_commission_reserve,
                    "free_funds": account_free_funds
                },
                "bets": []
            }
        
        # Parse Matchbook response
        markets = current_bets_response.get('markets', [])
        
        bets = []
        total_stake = 0.0
        total_potential_profit = 0.0
        
        conn = duckdb.connect('football_data.duckdb')
        
        for market in markets:
            event_name = market.get('event-name', 'Unknown')
            market_name = market.get('market-name', 'Unknown')
            
            for selection in market.get('selections', []):
                runner_name = selection.get('runner-name', 'Unknown')
                runner_id = selection.get('runner-id')
                odds = selection.get('odds', 0)
                stake = selection.get('remaining-stake', 0) + selection.get('matched-stake', 0)
                matched_stake = selection.get('matched-stake', 0)
                remaining_stake = selection.get('remaining-stake', 0)
                potential_profit = selection.get('potential-profit', 0)
                potential_loss = selection.get('potential-loss', 0)
                
                # Calculate current P/L based on bet status
                # For now, use potential profit/loss from Matchbook
                current_pl = potential_profit if potential_profit > 0 else -abs(potential_loss)
                
                # Try to get original bet info from bet_history using runner_id
                try:
                    bet_history = conn.execute("""
                        SELECT 
                            match_name,
                            league,
                            market,
                            stake,
                            expected_value,
                            confidence,
                            placed_at
                        FROM bet_history
                        WHERE runner_id = ?
                        ORDER BY placed_at DESC
                        LIMIT 1
                    """, [runner_id]).fetchone()
                    
                    if bet_history:
                        match_name = bet_history[0]
                        league = bet_history[1]
                        market = bet_history[2]
                        db_stake = bet_history[3]
                        expected_value = bet_history[4]
                        confidence = bet_history[5]
                        placed_at = bet_history[6]
                        
                        # If Matchbook stake is 0, use stake from bet_history
                        if stake == 0 and db_stake > 0:
                            stake = db_stake
                    else:
                        match_name = event_name  # Fallback to Matchbook event name
                        league = "Unknown"
                        market = market_name
                        expected_value = 0
                        confidence = 0
                        placed_at = None
                except Exception:
                    match_name = event_name
                    league = "Unknown"
                    market = market_name
                    expected_value = 0
                    confidence = 0
                    placed_at = None
                
                # If stake is still 0, default to 0.10
                if stake == 0:
                    stake = 0.10
                
                # Calculate current EV (stake × expected_value)
                current_ev = stake * expected_value if expected_value > 0 else 0
                
                bets.append({
                    "runner_id": runner_id,
                    "event_name": event_name,
                    "match_name": match_name,  # From bet_history
                    "league": league,  # From bet_history
                    "market_name": market_name,
                    "market": market,  # From bet_history
                    "runner_name": runner_name,
                    "odds": odds,
                    "stake": stake,
                    "matched_stake": matched_stake,
                    "remaining_stake": remaining_stake,
                    "potential_profit": potential_profit,
                    "potential_loss": potential_loss,
                    "current_pl": current_pl,
                    "expected_value": expected_value,
                    "current_ev": current_ev,
                    "confidence": confidence,
                    "placed_at": placed_at
                })
                
                total_stake += stake
                total_potential_profit += potential_profit
        
        conn.close()
        
        # Calculate total current EV
        total_current_ev = sum(bet['current_ev'] for bet in bets)
        
        return {
            "total_bets": len(bets),
            "total_stake": total_stake,
            "potential_profit": total_potential_profit,
            "current_pl": total_pl,
            "total_current_ev": total_current_ev,
            "account": {
                "balance": account_balance,
                "exposure": account_exposure,
                "commission_reserve": account_commission_reserve,
                "free_funds": account_free_funds
            },
            "bets": bets
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting current bets: {str(e)}")

@app.get("/status")
async def get_status():
    """Get system status and statistics"""
    try:
        conn = duckdb.connect('football_data.duckdb')
        
        # Get database stats
        fixtures_count = conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
        predictions_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        odds_count = conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
        bets_count = conn.execute("SELECT COUNT(*) FROM bet_history").fetchone()[0]
        
        # Get cache stats
        cache_count = conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
        cache_expired = conn.execute(
            "SELECT COUNT(*) FROM api_cache WHERE expires_at < CURRENT_TIMESTAMP"
        ).fetchone()[0]
        
        # Get recent bets stats
        today_bets = conn.execute("""
            SELECT COUNT(*), COALESCE(SUM(stake), 0)
            FROM bet_history
            WHERE placed_at >= CURRENT_DATE
        """).fetchone()
        
        week_bets = conn.execute("""
            SELECT COUNT(*), COALESCE(SUM(stake), 0)
            FROM bet_history
            WHERE placed_at >= CURRENT_DATE - INTERVAL '7 days'
        """).fetchone()
        
        conn.close()
        
        return {
            "timestamp": datetime.now(),
            "workflow": {
                "is_running": workflow_status["is_running"],
                "last_run": workflow_status["last_run"]
            },
            "database": {
                "fixtures": fixtures_count,
                "predictions": predictions_count,
                "odds": odds_count,
                "total_bets": bets_count
            },
            "cache": {
                "total_entries": cache_count,
                "expired_entries": cache_expired
            },
            "betting": {
                "today": {
                    "count": today_bets[0],
                    "total_stake": float(today_bets[1])
                },
                "last_7_days": {
                    "count": week_bets[0],
                    "total_stake": float(week_bets[1])
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.post("/cleanup-cache")
async def cleanup_cache():
    """Clean up expired cache entries"""
    try:
        conn = duckdb.connect('football_data.duckdb')
        
        result = conn.execute("""
            DELETE FROM api_cache 
            WHERE expires_at < CURRENT_TIMESTAMP
        """)
        
        conn.close()
        
        return {
            "status": "success",
            "message": "Expired cache entries removed",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning cache: {str(e)}")

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
