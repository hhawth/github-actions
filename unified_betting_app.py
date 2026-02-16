import streamlit as st
import pandas as pd
import duckdb
from datetime import datetime
from quantitative_betting_workflow import QuantitativeBettingWorkflow
import sys
from io import StringIO
import time

# Initialize database sync on startup
@st.cache_resource
def initialize_database_sync():
    """Initialize database sync on app startup"""
    try:
        from database_sync import ensure_database_exists, start_periodic_upload
        
        # Download database on startup
        if ensure_database_exists():
            st.success("✅ Database ready")
            
            # Start periodic upload to GCS (every hour)
            upload_thread = start_periodic_upload(daemon=True)
            st.info("🔄 Periodic GCS upload enabled (hourly)")
            return True
        else:
            st.warning("⚠️ Database setup failed, proceeding anyway")
            return False
    except Exception as e:
        st.error(f"❌ Database sync initialization failed: {e}")
        return False

# Initialize database sync
db_sync_status = initialize_database_sync()

# Run workflow on startup and cache results
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_workflow_cached():
    """Run the betting workflow and cache results for 30 minutes"""
    
    # Capture output
    output_capture = StringIO()
    sys.stdout = output_capture
    
    start_time = time.time()
    
    try:
        # Initialize and run workflow
        workflow = QuantitativeBettingWorkflow({
            'min_ev': 0.08,
            'min_confidence': 0.65,
            'max_daily_stake': 5.0,
            'auto_place_bets': True,
            'target_markets': ['btts', 'over_under_15', 'handicap']
        })
        
        # Run the full workflow
        workflow.run_full_workflow()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        duration = time.time() - start_time
        
        return {
            'status': 'success',
            'duration': duration,
            'log': output_capture.getvalue(),
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        sys.stdout = sys.__stdout__
        return {
            'status': 'error',
            'error': str(e),
            'log': output_capture.getvalue(),
            'timestamp': datetime.now()
        }

# Page config
st.set_page_config(
    page_title="Quantitative Betting Dashboard", 
    page_icon="⚽",
    layout="wide"
)

# Title
st.title("⚽ Quantitative Betting Dashboard")

# Show database sync status
if db_sync_status:
    st.success("🔄 Database sync enabled - Connected to GCS")
else:
    st.warning("⚠️ Database sync disabled - Running locally only")

# Run workflow on app startup (cached)
workflow_result = run_workflow_cached()

# Show workflow status
if workflow_result['status'] == 'success':
    st.success(f"✅ Workflow completed in {workflow_result['duration']:.1f}s at {workflow_result['timestamp'].strftime('%H:%M:%S')}")
else:
    st.error(f"❌ Workflow failed: {workflow_result.get('error', 'Unknown error')}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["💰 Current Bets", "📊 Bet History", "📋 Workflow Log"])

with tab1:
    st.header("💰 Current Bets")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_current"):
            st.cache_data.clear()  # Clear cache to force refresh
            st.rerun()
    
    try:
        # Connect to database directly
        conn = duckdb.connect('football_data.duckdb')
        
        # Get recent bets from last 24 hours
        recent_bets = conn.execute("""
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
            WHERE placed_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
            ORDER BY placed_at DESC
        """).df()
        
        if len(recent_bets) > 0:
            st.subheader("📊 Recent Bets (Last 24h)")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Bets", len(recent_bets))
            with col2:
                st.metric("Total Stake", f"£{recent_bets['stake'].sum():.2f}")
            with col3:
                st.metric("Avg EV", f"{recent_bets['expected_value'].mean()*100:.1f}%")
            with col4:
                total_ev_profit = (recent_bets['stake'] * recent_bets['expected_value']).sum()
                st.metric("Total EV Profit", f"£{total_ev_profit:.2f}")
            
            # Display bets
            display_df = recent_bets.copy()
            display_df['stake'] = display_df['stake'].apply(lambda x: f"£{x:.2f}")
            display_df['odds'] = display_df['odds'].apply(lambda x: f"{x:.2f}")
            display_df['expected_value'] = display_df['expected_value'].apply(lambda x: f"{x*100:.1f}%")
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
            display_df['placed_at'] = pd.to_datetime(display_df['placed_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No bets placed in the last 24 hours")
            st.write("The workflow runs automatically when the app starts. Check the 'Workflow Log' tab for details.")
            
        conn.close()
        
    except Exception as e:
        st.error(f"❌ Error loading current bets: {str(e)}")

with tab2:
    st.header("📊 Bet History")
    
    # Time filter
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        days_back = st.selectbox("Period", [1, 7, 30, 90], index=1)
    with col2:
        limit = st.selectbox("Max bets", [10, 50, 100, 500], index=1)
    
    try:
        # Connect to database directly
        conn = duckdb.connect('football_data.duckdb')
        
        # Get bet history from database 
        bet_history = conn.execute(f"""
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
            WHERE placed_at >= CURRENT_TIMESTAMP - INTERVAL '{days_back} days'
            ORDER BY placed_at DESC
            LIMIT {limit}
        """).df()
        
        if len(bet_history) > 0:
            # Summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Bets", len(bet_history))
            with col2:
                st.metric("Total Stake", f"£{bet_history['stake'].sum():.2f}")
            with col3:
                st.metric("Avg Odds", f"{bet_history['odds'].mean():.2f}")
            with col4:
                st.metric("Avg EV", f"{bet_history['expected_value'].mean()*100:.1f}%")
            with col5:
                # Calculate total EV profit: sum of (stake * expected_value) for each bet
                total_ev_profit = (bet_history['stake'] * bet_history['expected_value']).sum()
                st.metric("Total EV Profit", f"£{total_ev_profit:.2f}")
            
            st.divider()
            
            # Format display
            display_df = bet_history.copy()
            display_df['stake'] = display_df['stake'].apply(lambda x: f"£{x:.2f}")
            display_df['odds'] = display_df['odds'].apply(lambda x: f"{x:.2f}")
            display_df['expected_value'] = display_df['expected_value'].apply(lambda x: f"{x*100:.1f}%")
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
            display_df['placed_at'] = pd.to_datetime(display_df['placed_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "runner_id": "Runner ID",
                    "match_name": "Match",
                    "league": "League",
                    "market": "Market",
                    "runner_name": "Selection",
                    "stake": "Stake",
                    "odds": "Odds",
                    "expected_value": "EV",
                    "confidence": "Confidence",
                    "placed_at": "Placed At"
                }
            )
        else:
            st.info(f"No bets placed in the last {days_back} days")
            
        conn.close()
            
    except Exception as e:
        st.error(f"❌ Error loading bet history: {str(e)}")

with tab3:
    st.header("📋 Workflow Execution Log")
    
    # Show workflow execution details
    if workflow_result['status'] == 'success':
        st.success(f"✅ Workflow completed successfully in {workflow_result['duration']:.1f} seconds")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Execution Time", f"{workflow_result['duration']:.1f}s")
        with col2:
            st.metric("Last Run", workflow_result['timestamp'].strftime('%H:%M:%S'))
        
    else:
        st.error(f"❌ Workflow failed: {workflow_result.get('error', 'Unknown error')}")
    
    st.divider()
    
    # Configuration used
    st.subheader("⚙️ Configuration")
    st.write("**Current Settings:**")
    st.write("- Minimum EV: **8%**")
    st.write("- Minimum Confidence: **65%**") 
    st.write("- Max Daily Stake: **£5.00**")
    st.write("- Auto-place bets: **✅ Yes**")
    st.write("- Target Markets: **BTTS, Over/Under 1.5, Handicap**")
    
    st.divider()
    
    # Full execution log
    st.subheader("📜 Execution Log")
    if workflow_result.get('log'):
        with st.expander("View Full Log", expanded=False):
            st.text(workflow_result['log'])
    else:
        st.info("No log available")
    
    # Manual refresh option
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Force Refresh Cache", type="secondary"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        st.caption("Cache refreshes automatically every hour")

# Footer
st.divider()
sync_status = "GCS Sync ✅" if db_sync_status else "Local Only ⚠️"
st.caption(f"🚀 Auto-workflow enabled • {sync_status} • Cache: 30min • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
