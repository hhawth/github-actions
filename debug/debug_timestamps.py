#!/usr/bin/env python3
"""
Debug Timestamp Conversion
"""

from datetime import datetime

def test_timestamp_conversion():
    print("⏱️ TESTING TIMESTAMP CONVERSION")
    print("=" * 40)
    
    # Test Matchbook time format
    matchbook_time = "2026-02-13T10:30:00.000Z"
    prediction_timestamp = 1770940800
    
    print(f"Matchbook time: {matchbook_time}")
    print(f"Prediction timestamp: {prediction_timestamp}")
    
    try:
        # Parse matchbook time (ISO format) 
        mb_dt = datetime.fromisoformat(matchbook_time.replace('Z', '+00:00'))
        mb_timestamp = mb_dt.timestamp()
        
        print(f"Matchbook timestamp: {mb_timestamp}")
        print(f"Time difference: {abs(prediction_timestamp - mb_timestamp)} seconds")
        print(f"Time difference: {abs(prediction_timestamp - mb_timestamp) / 3600:.2f} hours")
        
        # Check if within 30 minutes (1800 seconds)
        within_30_min = abs(prediction_timestamp - mb_timestamp) <= 1800
        print(f"Within 30 minutes: {within_30_min}")
        
        # Convert timestamps back to readable times
        pred_dt = datetime.fromtimestamp(prediction_timestamp)
        print("\nReadable times:")
        print(f"Matchbook: {mb_dt}")  
        print(f"Prediction: {pred_dt}")
        
    except Exception as e:
        print(f"Error parsing timestamp: {e}")

if __name__ == "__main__":
    test_timestamp_conversion()