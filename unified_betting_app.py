import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import threading
import time
import uvicorn
from api_server import app

# Start FastAPI in background thread
@st.cache_resource
def start_api_server():
    """Start FastAPI server in background thread"""
    def run_server():
        port = 8080
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    time.sleep(2)  # Wait for server to start
    return "http://localhost:8080"

# Initialize API
API_URL = start_api_server()

# Page config
st.set_page_config(
    page_title="Quantitative Betting Dashboard",
    page_icon="⚽",
    layout="wide"
)

# Title
st.title("⚽ Quantitative Betting Dashboard")
st.caption(f"API Status: {API_URL}")

# Default configuration (no sidebar)
min_ev = 8
min_confidence = 65
max_daily_stake = 5.0
auto_bet = True

# Create tabs
tab1, tab2, tab3 = st.tabs(["💰 Current Bets", "📊 Bet History", "🎯 Run Workflow"])

with tab1:
    st.header("💰 Current Bets")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_current"):
            st.rerun()
    
    try:
        response = requests.get(f"{API_URL}/current-bets", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Account Information
            if data.get('account'):
                account = data['account']
                st.subheader("💳 Account Status")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Balance", f"£{account['balance']:.2f}")
                with col2:
                    st.metric("Exposure", f"£{account['exposure']:.2f}")
                with col3:
                    st.metric("Commission Reserve", f"£{account['commission_reserve']:.2f}")
                with col4:
                    st.metric("Free Funds", f"£{account['free_funds']:.2f}")
                
                st.divider()
            
            # Betting Summary
            st.subheader("📊 Active Bets Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Active Bets", data['total_bets'])
            with col2:
                st.metric("Total Stake", f"£{data['total_stake']:.2f}")
            with col3:
                current_ev = data.get('total_current_ev', 0)
                st.metric("Total EV", f"£{current_ev:.2f}")
            with col4:
                potential = data['potential_profit']
                st.metric("Potential Profit", f"£{potential:.2f}")
            with col5:
                pl = data['current_pl']
                st.metric(
                    "Current P/L",
                    f"£{pl:.2f}",
                    delta=f"{pl:.2f}",
                    delta_color="normal" if pl >= 0 else "inverse"
                )
            
            st.divider()
            
            if data['total_bets'] > 0:
                bets = data['bets']
                
                # Display each bet
                for bet in bets:
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            # Use match_name from bet_history if available
                            match_display = bet.get('match_name', bet['event_name'])
                            st.write(f"**{match_display}**")
                            
                            # Show league if available
                            if bet.get('league') and bet['league'] != 'Unknown':
                                st.caption(f"🏆 {bet['league']}")
                            
                            # Use market from bet_history if available
                            market_display = bet.get('market', bet['market_name'])
                            st.write(f"📊 {market_display} - {bet['runner_name']}")
                            
                            if bet.get('placed_at'):
                                st.caption(f"🕐 Placed: {bet['placed_at']}")
                        
                        with col2:
                            st.write(f"**Odds:** {bet['odds']:.2f}")
                            st.write(f"**Stake:** £{bet['stake']:.2f}")
                            if bet.get('expected_value', 0) > 0:
                                st.write(f"**EV:** {bet['expected_value']*100:.1f}%")
                                current_ev = bet.get('current_ev', 0)
                                if current_ev > 0:
                                    st.write(f"**Current EV:** £{current_ev:.2f}")
                            if bet.get('confidence', 0) > 0:
                                st.write(f"**Confidence:** {bet['confidence']*100:.1f}%")
                        
                        with col3:
                            matched = bet['matched_stake']
                            remaining = bet['remaining_stake']
                            
                            if matched > 0:
                                st.success(f"✅ Matched: £{matched:.2f}")
                            if remaining > 0:
                                st.warning(f"⏳ Pending: £{remaining:.2f}")
                            
                            # P/L indicator
                            pl = bet['current_pl']
                            if pl > 0:
                                st.success(f"📈 P/L: +£{pl:.2f}")
                            elif pl < 0:
                                st.error(f"📉 P/L: -£{abs(pl):.2f}")
                            else:
                                st.info("P/L: £0.00")
                        
                        st.divider()
                
                # Summary footer
                st.write("**Summary:**")
                avg_ev = sum(b['expected_value'] for b in bets if b['expected_value'] > 0) / max(len([b for b in bets if b['expected_value'] > 0]), 1)
                st.write(f"- Average EV: {avg_ev*100:.1f}%")
                st.write(f"- Total Potential Return: £{data['total_stake'] + data['potential_profit']:.2f}")
                roi = (data['current_pl'] / data['total_stake'] * 100) if data['total_stake'] > 0 else 0
                st.write(f"- Current ROI: {roi:.1f}%")
            else:
                st.info("📋 No active bets")
                st.write("Place bets through the 'Run Workflow' tab to see them here.")
        else:
            st.error("Failed to load current bets")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.write("Make sure the Matchbook API is configured and accessible.")

with tab2:
    st.header("Bet History")
    
    # Time filter
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        days_back = st.selectbox("Period", [1, 7, 30, 90], index=1)
    with col2:
        limit = st.selectbox("Max bets", [10, 50, 100, 500], index=1)
    
    if st.button("🔄 Refresh", key="refresh_history"):
        st.rerun()
    
    try:
        response = requests.get(
            f"{API_URL}/bet-history",
            params={"days": days_back, "limit": limit},
            timeout=5
        )
        
        if response.status_code == 200:
            bets = response.json()
            
            if len(bets) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(bets)
                
                # Summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Bets", len(bets))
                with col2:
                    st.metric("Total Stake", f"£{df['stake'].sum():.2f}")
                with col3:
                    st.metric("Avg Odds", f"{df['odds'].mean():.2f}")
                with col4:
                    st.metric("Avg EV", f"{df['expected_value'].mean()*100:.1f}%")
                with col5:
                    # Calculate total EV profit: sum of (stake * expected_value) for each bet
                    total_ev_profit = (df['stake'] * df['expected_value']).sum()
                    st.metric("Total EV Profit", f"£{total_ev_profit:.2f}")
                
                st.divider()
                
                # Format display
                display_df = df.copy()
                display_df['stake'] = display_df['stake'].apply(lambda x: f"£{x:.2f}")
                display_df['odds'] = display_df['odds'].apply(lambda x: f"{x:.2f}")
                display_df['expected_value'] = display_df['expected_value'].apply(lambda x: f"{x*100:.1f}%")
                display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
                display_df['placed_at'] = pd.to_datetime(display_df['placed_at']).dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(
                    display_df,
                    width="stretch",
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
        else:
            st.error("Failed to load bet history")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

with tab3:
    st.header("Run Betting Workflow")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        st.write(f"- Minimum EV: **{min_ev}%**")
        st.write(f"- Minimum Confidence: **{min_confidence}%**")
        st.write(f"- Max Daily Stake: **£{max_daily_stake:.2f}**")
        st.write(f"- Auto-place bets: **{'✅ Yes' if auto_bet else '❌ No'}**")
        
        if auto_bet:
            st.warning("⚠️ Auto-betting is ENABLED. Real bets will be placed!")
        else:
            st.info("ℹ️ Dry-run mode. No real bets will be placed.")
    
    with col2:
        st.subheader("Actions")
        
        if st.button("🚀 Run Workflow Now", type="primary", width="stretch"):
            with st.spinner("Starting workflow..."):
                try:
                    payload = {
                        "min_ev_threshold": min_ev / 100,
                        "min_confidence": min_confidence / 100,
                        "max_daily_stake": max_daily_stake,
                        "auto_place_bets": auto_bet
                    }
                    
                    response = requests.post(
                        f"{API_URL}/run-workflow",
                        json=payload,
                        timeout=60  # Increase timeout to 60 seconds for workflow completion
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.json(result)
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Request failed: {str(e)}")
        
        if st.button("📋 Check Workflow Status", width="stretch"):
            try:
                response = requests.get(f"{API_URL}/workflow-status", timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    
                    if status['is_running']:
                        st.info("🔄 Workflow is currently running...")
                    else:
                        st.write("**Last Run:**", status.get('last_run', 'Never'))
                        
                        if status.get('last_result'):
                            result = status['last_result']
                            if result.get('status') == 'success':
                                st.success("✅ Last run successful")
                            else:
                                st.error(f"❌ Last run failed: {result.get('error')}")
                            
                            if result.get('log'):
                                with st.expander("📄 View Log"):
                                    st.text(result['log'])
                else:
                    st.error("Failed to get status")
                    
            except Exception as e:
                st.error(f"❌ Request failed: {str(e)}")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | API: {API_URL}")
