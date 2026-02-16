import streamlit as st
import pandas as pd
import duckdb
from datetime import datetime
from quantitative_betting_workflow import QuantitativeBettingWorkflow
import sys
from io import StringIO

# Page config
st.set_page_config(
    page_title="Quantitative Betting Dashboard",
    page_icon="⚽",
    layout="wide"
)

# Title
st.title("⚽ Quantitative Betting Dashboard")

# Sidebar configuration
st.sidebar.header("Configuration")
min_ev = st.sidebar.slider("Minimum EV %", 0, 30, 8, 1)
min_confidence = st.sidebar.slider("Minimum Confidence %", 0, 100, 65, 5)
max_daily_stake = st.sidebar.number_input("Max Daily Stake (£)", min_value=1.0, max_value=100.0, value=5.0, step=1.0)
auto_bet = st.sidebar.checkbox("Auto-place bets", value=False)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Opportunities", "📊 Bet History", "⚙️ System Info"])

with tab1:
    st.header("Current Betting Opportunities")
    
    if st.button("🔄 Refresh Opportunities", type="primary"):
        with st.spinner("Analyzing matches and finding opportunities..."):
            # Capture output
            output_capture = StringIO()
            sys.stdout = output_capture
            
            try:
                # Initialize workflow
                workflow = QuantitativeBettingWorkflow(
                    min_ev_threshold=min_ev / 100,
                    min_confidence=min_confidence / 100,
                    max_daily_stake=max_daily_stake,
                    auto_place_bets=auto_bet
                )
                
                # Run workflow
                workflow.run_full_workflow()
                
                # Restore stdout
                sys.stdout = sys.__stdout__
                
                # Show output in expander
                with st.expander("📋 View Detailed Log"):
                    st.text(output_capture.getvalue())
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                sys.stdout = sys.__stdout__
                st.error(f"❌ Error: {str(e)}")
                st.text(output_capture.getvalue())
    
    # Load opportunities from recent analysis
    st.subheader("Available Opportunities")
    
    conn = duckdb.connect('football_data.duckdb')
    
    # Get today's fixtures with predictions
    try:
        fixtures_df = conn.execute("""
            SELECT 
                f.fixture_id,
                f.date,
                f.league_name,
                f.home_team_name,
                f.away_team_name,
                p.home_team_prediction,
                p.away_team_prediction
            FROM fixtures f
            JOIN predictions p ON f.fixture_id = p.fixture_id
            WHERE f.date >= CURRENT_DATE
            AND f.date <= CURRENT_DATE + INTERVAL '7 days'
            ORDER BY f.date
            LIMIT 50
        """).df()
        
        if len(fixtures_df) > 0:
            st.write(f"Found {len(fixtures_df)} upcoming matches with predictions")
            
            # Display in a nice format
            for idx, row in fixtures_df.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([2, 3, 2])
                    
                    with col1:
                        st.write(f"**{row['league_name']}**")
                        st.write(f"📅 {row['date']}")
                    
                    with col2:
                        st.write(f"### {row['home_team_name']} vs {row['away_team_name']}")
                    
                    with col3:
                        if st.button("📊 View Details", key=f"detail_{row['fixture_id']}"):
                            st.write(f"Predictions for fixture {row['fixture_id']}")
                    
                    st.divider()
        else:
            st.info("No upcoming matches found with predictions")
            
    except Exception as e:
        st.warning(f"Could not load fixtures: {str(e)}")
    
    conn.close()

with tab2:
    st.header("Bet History")
    
    conn = duckdb.connect('football_data.duckdb')
    
    # Time filter
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.selectbox("Show bets from", [1, 7, 30, 90, 365], index=1)
    
    try:
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
        """).df()
        
        if len(bet_history) > 0:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Bets", len(bet_history))
            with col2:
                st.metric("Total Stake", f"£{bet_history['stake'].sum():.2f}")
            with col3:
                avg_odds = bet_history['odds'].mean()
                st.metric("Avg Odds", f"{avg_odds:.2f}")
            with col4:
                avg_ev = bet_history['expected_value'].mean() * 100
                st.metric("Avg EV", f"{avg_ev:.1f}%")
            
            st.divider()
            
            # Format the dataframe
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
            
    except Exception as e:
        st.warning(f"Could not load bet history: {str(e)}")
    
    conn.close()

with tab3:
    st.header("System Information")
    
    conn = duckdb.connect('football_data.duckdb')
    
    # Database stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Database Statistics")
        try:
            fixtures_count = conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
            st.metric("Total Fixtures", f"{fixtures_count:,}")
            
            predictions_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            st.metric("Total Predictions", f"{predictions_count:,}")
            
            odds_count = conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
            st.metric("Total Odds Records", f"{odds_count:,}")
            
            bets_count = conn.execute("SELECT COUNT(*) FROM bet_history").fetchone()[0]
            st.metric("Total Bets Placed", f"{bets_count:,}")
            
        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")
    
    with col2:
        st.subheader("🔧 Configuration")
        st.write("**Current Settings:**")
        st.write(f"- Minimum EV: {min_ev}%")
        st.write(f"- Minimum Confidence: {min_confidence}%")
        st.write(f"- Max Daily Stake: £{max_daily_stake:.2f}")
        st.write(f"- Auto-place bets: {'✅ Yes' if auto_bet else '❌ No'}")
        
        st.divider()
        
        st.write("**API Cache:**")
        try:
            cache_count = conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
            st.write(f"- Cached entries: {cache_count}")
            
            expired = conn.execute("SELECT COUNT(*) FROM api_cache WHERE expires_at < CURRENT_TIMESTAMP").fetchone()[0]
            st.write(f"- Expired entries: {expired}")
            
            if st.button("🗑️ Clear Expired Cache"):
                conn.execute("DELETE FROM api_cache WHERE expires_at < CURRENT_TIMESTAMP")
                st.success("Cache cleaned!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error checking cache: {str(e)}")
    
    conn.close()
    
    # Recent activities
    st.subheader("📝 Recent Activity")
    conn = duckdb.connect('football_data.duckdb')
    try:
        recent_bets = conn.execute("""
            SELECT 
                match_name,
                market,
                stake,
                odds,
                placed_at
            FROM bet_history
            ORDER BY placed_at DESC
            LIMIT 5
        """).df()
        
        if len(recent_bets) > 0:
            for idx, bet in recent_bets.iterrows():
                st.write(f"🎯 {bet['match_name']} - {bet['market']} @ {bet['odds']:.2f} (£{bet['stake']:.2f}) - {bet['placed_at']}")
        else:
            st.info("No recent bets")
            
    except Exception as e:
        st.warning(f"Could not load recent activity: {str(e)}")
    
    conn.close()

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
