#!/usr/bin/env python3
"""
Launcher script for the Streamlit Multi-Agent Football Prediction System
"""

import subprocess
import sys

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import plotly
        print("✅ Required packages found")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install with: pip install streamlit plotly")
        return False

def launch_app():
    """Launch the Streamlit app"""
    if not check_requirements():
        return False
    
    try:
        print("🚀 Launching Multi-Agent Football Prediction System...")
        print("📱 App will open in your browser at http://localhost:8501")
        print("🛑 Press Ctrl+C to stop the server")
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("⚽ Multi-Agent Football Prediction System")
    print("=" * 50)
    
    launch_app()