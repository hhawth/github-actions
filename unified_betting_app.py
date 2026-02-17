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
    """Initialize database sync on app startup - returns status dict"""
    try:
        from database_sync import ensure_database_exists
        
        # Download database on startup
        if ensure_database_exists():
            return {
                'status': 'success',
                'message': 'Database ready',
                'upload_enabled': False  # Will upload after workflow runs
            }
        else:
            return {
                'status': 'error',
                'message': 'Could not initialize database',
                'details': 'Cannot start application without valid database from GCS'
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': 'Database sync initialization failed',
            'details': str(e)
        }

# Initialize database sync and handle UI
db_sync_result = initialize_database_sync()

if db_sync_result['status'] == 'success':
    st.toast("✅ Database ready", icon="✅")
elif db_sync_result['status'] == 'error':
    st.error(f"❌ CRITICAL: {db_sync_result['message']}")
    if db_sync_result.get('details'):
        st.error(f"❌ {db_sync_result['details']}")
    st.stop()

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
            'min_stake': 0.10,
            'max_daily_stake': 5.0,
            'max_bets_per_match': 1,
            'auto_place_bets': True,
            'target_markets': ['btts', 'over_under_15', 'handicap']
        })
        
        # Run the full workflow
        workflow.run_full_workflow()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        duration = time.time() - start_time
        
        # Upload updated database to GCS after successful workflow
        try:
            from pathlib import Path
            db_file = Path("football_data.duckdb")
            
            if db_file.exists():
                size_mb = db_file.stat().st_size / (1024 * 1024)
                print(f"🔧 Debug: Database size after workflow: {size_mb:.2f} MB")
                
                # Only upload if database is reasonably large (>50MB)
                if size_mb > 50:
                    from database_sync import upload_to_gcs
                    upload_success = upload_to_gcs()
                    if upload_success:
                        print("✅ Database uploaded to GCS after workflow completion")
                    else:
                        print("⚠️ Warning: Database upload to GCS failed")
                else:
                    print(f"🔧 Debug: Database too small ({size_mb:.2f}MB), skipping upload to prevent overwriting good database")
            else:
                print("⚠️ Warning: No database file found after workflow, skipping upload")
        except Exception as upload_error:
            print(f"⚠️ Warning: Database upload error: {upload_error}")
        
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
if db_sync_result['status'] == 'success':
    st.toast("🔄 Database sync enabled - Uploads after workflow runs", icon="🔄")
else:
    st.warning("⚠️ Database sync disabled - Running locally only")

# Run workflow on app startup (cached)
workflow_result = run_workflow_cached()

# Show workflow status
if workflow_result['status'] == 'success':
    st.toast(f"✅ Workflow completed in {workflow_result['duration']:.1f}s at {workflow_result['timestamp'].strftime('%H:%M:%S')}", icon="✅")
else:
    st.error(f"❌ Workflow failed: {workflow_result.get('error', 'Unknown error')}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["💰 Current Bets", "📊 Bet History", "📋 Workflow Log"])

with tab1:
    st.header("🎯 Current Active Bets")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_current"):
            st.cache_data.clear()  # Clear cache to force refresh
            st.rerun()
    
    try:
        # Get live bets from Matchbook and match with database
        from matchbook import matchbookExchange
        
        with st.spinner("🏢 Fetching live bets from Matchbook..."):
            matchbook = matchbookExchange()
            matchbook.login()
            
            # Get current active bets and account info
            current_bets_response = matchbook.get_current_bets()
            account_data = matchbook.get_account()
            
        if not current_bets_response or not current_bets_response.get('markets'):
            # No active bets, show recent database bets as fallback
            st.info("No active bets found on Matchbook")
            
            conn = duckdb.connect('football_data.duckdb')
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
            conn.close()
            
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
                
        else:
            # Process live Matchbook bets and match with database
            markets = current_bets_response.get('markets', [])
            
            # Account summary
            balance = account_data.get('balance', 0)
            exposure = account_data.get('exposure', 0)
            free_funds = account_data.get('free-funds', 0)
            
            st.subheader("💰 Account Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Balance", f"£{balance:.2f}")
            with col2:
                st.metric("Exposure", f"£{exposure:.2f}")
            with col3:
                st.metric("Free Funds", f"£{free_funds:.2f}")
            
            # Process active bets
            conn = duckdb.connect('football_data.duckdb')
            
            active_bets = []
            total_stake = 0.0
            total_potential_profit = 0.0
            
            for market in markets:
                event_name = market.get('event-name', 'Unknown')
                market_name = market.get('market-name', 'Unknown')
                
                for selection in market.get('selections', []):
                    runner_id = selection.get('runner-id')
                    runner_name = selection.get('runner-name', 'Unknown')
                    odds = selection.get('odds', 0)
                    matched_stake = selection.get('matched-stake', 0)
                    remaining_stake = selection.get('remaining-stake', 0)
                    total_bet_stake = matched_stake + remaining_stake
                    potential_profit = selection.get('potential-profit', 0)
                    potential_loss = selection.get('potential-loss', 0)
                    
                    # Get original bet info from database using runner_id
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
                            market_type = bet_history[2]
                            db_stake = bet_history[3]
                            expected_value = bet_history[4]
                            confidence = bet_history[5]
                            placed_at = bet_history[6]
                            
                            # Use database stake if Matchbook shows 0
                            if total_bet_stake == 0 and db_stake > 0:
                                total_bet_stake = db_stake
                        else:
                            # Fallback to Matchbook data
                            match_name = event_name
                            league = "Unknown"
                            market_type = market_name
                            expected_value = 0
                            confidence = 0
                            placed_at = None
                            if total_bet_stake == 0:
                                total_bet_stake = 0.10
                    except Exception:
                        match_name = event_name
                        league = "Unknown"
                        market_type = market_name
                        expected_value = 0
                        confidence = 0
                        placed_at = None
                        if total_bet_stake == 0:
                            total_bet_stake = 0.10
                    
                    # Calculate current EV profit
                    current_ev_profit = total_bet_stake * expected_value if expected_value > 0 else 0
                    
                    active_bets.append({
                        'runner_id': runner_id,
                        'match_name': match_name,
                        'league': league,
                        'market': market_type,
                        'runner_name': runner_name,
                        'odds': odds,
                        'stake': total_bet_stake,
                        'matched_stake': matched_stake,
                        'remaining_stake': remaining_stake,
                        'potential_profit': potential_profit,
                        'potential_loss': potential_loss,
                        'expected_value': expected_value,
                        'current_ev_profit': current_ev_profit,
                        'confidence': confidence,
                        'placed_at': placed_at
                    })
                    
                    total_stake += total_bet_stake
                    total_potential_profit += potential_profit
            
            conn.close()
            
            if active_bets:
                st.subheader("🎯 Live Active Bets")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Active Bets", len(active_bets))
                with col2:
                    st.metric("Total Stake", f"£{total_stake:.2f}")
                with col3:
                    st.metric("Potential Profit", f"£{total_potential_profit:.2f}")
                with col4:
                    total_ev_profit = sum(bet['current_ev_profit'] for bet in active_bets)
                    st.metric("Expected EV Profit", f"£{total_ev_profit:.2f}")
                
                # Create DataFrame for display
                df = pd.DataFrame(active_bets)
                display_df = df.copy()
                
                # Format columns
                display_df['stake'] = display_df['stake'].apply(lambda x: f"£{x:.2f}")
                display_df['odds'] = display_df['odds'].apply(lambda x: f"{x:.2f}")
                display_df['potential_profit'] = display_df['potential_profit'].apply(lambda x: f"£{x:.2f}")
                display_df['expected_value'] = display_df['expected_value'].apply(lambda x: f"{x*100:.1f}%" if x > 0 else "N/A")
                display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%" if x > 0 else "N/A")
                display_df['current_ev_profit'] = display_df['current_ev_profit'].apply(lambda x: f"£{x:.2f}")
                
                # Select columns to show
                show_columns = ['match_name', 'league', 'market', 'runner_name', 'odds', 'stake', 
                              'potential_profit', 'expected_value', 'confidence', 'current_ev_profit']
                display_df = display_df[show_columns]
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No active bets found")
            
    except Exception as e:
        st.error(f"❌ Error loading current bets: {str(e)}")
        st.write("Falling back to recent database bets...")
        
        # Fallback to database bets
        try:
            conn = duckdb.connect('football_data.duckdb')
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
            conn.close()
            
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
        except Exception as fallback_error:
            st.error(f"❌ Fallback also failed: {str(fallback_error)}")

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
        st.toast(f"✅ Workflow completed successfully in {workflow_result['duration']:.1f} seconds", icon="✅")
        
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
sync_status = "GCS Sync ✅" if db_sync_result['status'] == 'success' else "Local Only ⚠️"
st.caption(f"🚀 Auto-workflow enabled • {sync_status} • Cache: 30min • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
