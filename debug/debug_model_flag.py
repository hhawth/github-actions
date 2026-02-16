#!/usr/bin/env python3
"""
Debug Model Training Flag
=========================
Check why is_trained flag is False even after model loading
"""

import sys
sys.path.append('.')

from improved_ev_betting_model import ImprovedEVBettingModel
import pickle
import os

def debug_model_training_flag():
    """Debug the is_trained flag issue"""
    
    print("🔍 DEBUGGING MODEL TRAINING FLAG")
    print("="*50)
    
    # Check if model file exists
    model_files = [f for f in os.listdir('.') if f.startswith('ev_model_') and f.endswith('.pkl')]
    
    if not model_files:
        print("❌ No model files found!")
        return
        
    latest_model = max(model_files)
    print(f"📁 Latest model file: {latest_model}")
    
    # Try to load the model file directly to check contents
    try:
        print("\n🔍 Loading model file directly...")
        with open(latest_model, 'rb') as f:
            model_data = pickle.load(f)
        
        print("✅ Model file loaded successfully")
        print(f"   📊 Keys in model file: {list(model_data.keys())}")
        print(f"   🎯 is_trained flag: {model_data.get('is_trained', 'NOT FOUND')}")
        print(f"   📈 Training stats: {model_data.get('training_stats', {}).get('match_result_cv_accuracy', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error loading model file: {e}")

    # Now try the ImprovedEVBettingModel class
    print("\n🤖 Testing ImprovedEVBettingModel...")
    try:
        model = ImprovedEVBettingModel()
        
        print("✅ Model initialized")
        print(f"   🎯 is_trained flag: {model.is_trained}")
        print(f"   📁 Model will try to load: {latest_model}")
        
        # Force a model load attempt
        if hasattr(model, '_load_model_if_exists'):
            print("\n🔄 Attempting to load model...")
            model._load_model_if_exists()
            print(f"   🎯 is_trained flag after load: {model.is_trained}")
        
    except Exception as e:
        print(f"❌ Error with ImprovedEVBettingModel: {e}")
        import traceback
        traceback.print_exc()

    # Check if the flag gets set during normal initialization
    print("\n🔬 DIAGNOSIS:")
    if model_data.get('is_trained') and not model.is_trained:
        print("🚨 ISSUE: Model file has is_trained=True but class has is_trained=False")
        print("💡 SOLUTION: The model loading is not happening or failing silently")
    elif model_data.get('is_trained') and model.is_trained:
        print("✅ Model loading works correctly")
    else:
        print("⚠️ Model file itself has is_trained=False - model may not be properly trained")

if __name__ == "__main__":
    debug_model_training_flag()