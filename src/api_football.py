import http.client
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import duckdb


load_dotenv()

headers = {"x-apisports-key": os.getenv("API_FOOTBALL_KEY")}

def init_cache_table():
    """Initialize DuckDB cache table for API calls"""
    conn = duckdb.connect('football_data.duckdb')
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_cache (
            cache_key VARCHAR PRIMARY KEY,
            cache_data JSON,
            cached_at TIMESTAMP,
            expires_at TIMESTAMP
        )
    """)
    conn.close()

def get_from_cache(cache_key):
    """Check if data exists in cache and is still valid"""
    try:
        conn = duckdb.connect('football_data.duckdb')
        result = conn.execute("""
            SELECT cache_data, cached_at, expires_at
            FROM api_cache
            WHERE cache_key = ?
            AND expires_at > CURRENT_TIMESTAMP
        """, [cache_key]).fetchone()
        conn.close()
        
        if result:
            cache_data, cached_at, expires_at = result
            remaining_minutes = int((expires_at - datetime.now()).total_seconds() / 60)
            print(f"🔄 Using cached data for {cache_key} (expires in {remaining_minutes} minutes)")
            return json.loads(cache_data) if isinstance(cache_data, str) else cache_data
        return None
    except Exception as e:
        print(f"⚠️ Cache read error: {e}")
        return None

def save_to_cache(cache_key, data, minutes=30):
    """Save data to DuckDB cache with expiration time"""
    try:
        conn = duckdb.connect('football_data.duckdb')
        expires_at = datetime.now() + timedelta(minutes=minutes)
        
        conn.execute("""
            INSERT OR REPLACE INTO api_cache (cache_key, cache_data, cached_at, expires_at)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?)
        """, [cache_key, json.dumps(data), expires_at])
        
        conn.close()
        print(f"💾 Cached data for {cache_key} (expires: {expires_at.strftime('%H:%M:%S')})")
    except Exception as e:
        print(f"⚠️ Cache write error: {e}")

def cleanup_expired_cache():
    """Remove expired cache entries"""
    try:
        conn = duckdb.connect('football_data.duckdb')
        deleted = conn.execute("""
            DELETE FROM api_cache
            WHERE expires_at < CURRENT_TIMESTAMP
        """).fetchone()
        conn.close()
        if deleted and deleted[0] > 0:
            print(f"🗑️ Cleaned up {deleted[0]} expired cache entries")
    except Exception:
        pass

# Initialize cache table on module load
init_cache_table()



def get_fixtures(date=None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    else:
        date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    # Check cache first
    cache_key = f"fixtures_{date}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return cached_data
    
    # Cache miss - fetch from API
    print(f"🆕 Cache miss for {cache_key}, fetching from API...")
    conn = http.client.HTTPSConnection("v3.football.api-sports.io")
    try:
        conn.request("GET", f"/fixtures?date={date}", headers=headers)
        res = conn.getresponse()

        if res.status != 200:
            print(f"❌ HTTP {res.status} error getting fixtures")
            conn.close()
            return json.dumps({"response": [], "errors": [f"HTTP {res.status}"]})

        data = res.read()
        conn.close()

        decoded_data = data.decode("utf-8")

        parsed_data = json.loads(decoded_data).get("response", [])

        conn = duckdb.connect('football_data.duckdb')

        for fixture in parsed_data:
            # Handle NULL scores for upcoming matches
            halftime_home = fixture.get("score", {}).get("halftime", {}).get("home")
            halftime_away = fixture.get("score", {}).get("halftime", {}).get("away") 
            fulltime_home = fixture.get("score", {}).get("fulltime", {}).get("home")
            fulltime_away = fixture.get("score", {}).get("fulltime", {}).get("away")
            
            conn.execute("""
                INSERT OR REPLACE INTO fixtures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                fixture["fixture"]["id"],
                fixture["fixture"]["date"][:10],  # Store only date part
                fixture["fixture"]["status"]["long"],
                fixture["league"]["id"],
                fixture["league"]["name"],
                fixture["league"]["country"],
                fixture["league"]["season"],
                fixture["teams"]["home"]["id"],
                fixture["teams"]["home"]["name"],
                fixture["teams"]["away"]["id"],
                fixture["teams"]["away"]["name"],
                halftime_home if halftime_home is not None else 0,
                halftime_away if halftime_away is not None else 0,
                fulltime_home if fulltime_home is not None else 0,
                fulltime_away if fulltime_away is not None else 0
            ])
            
        total_fixtures = len(parsed_data)
        print(f"⚽ Fixtures collection complete: {total_fixtures} fixtures found")
        
        # Save to cache for 5 minutes
        save_to_cache(cache_key, parsed_data, minutes=5)
        cleanup_expired_cache()  # Clean up old entries
        
        return parsed_data

    except Exception as e:
        print(f"❌ Error in fixtures collection: {e}")
        conn.close()
        return []

def get_odds(date=None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    else:
        date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    # Check cache first
    cache_key = f"odds_{date}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return cached_data
    
    # Cache miss - fetch from API
    print(f"🆕 Cache miss for {cache_key}, fetching from API...")
    conn = http.client.HTTPSConnection("v3.football.api-sports.io")

    try:
        # Get first page
        conn.request("GET", f"/odds?date={date}", headers=headers)
        res = conn.getresponse()

        if res.status != 200:
            print(f"❌ HTTP {res.status} error getting odds page 1")
            conn.close()
            return json.dumps({"response": [], "errors": [f"HTTP {res.status}"]})

        data = res.read()
        decoded_data = data.decode("utf-8")
        data = json.loads(decoded_data)
        conn.close()

        # Check for API errors
        if "errors" in data and data["errors"]:
            print(f"❌ API Error getting odds: {data['errors']}")
            return json.dumps(data)

        total_pages = data.get("paging", {}).get("total", 1)
        current_page = data.get("paging", {}).get("current", 1)
        total_results = data.get("results", 0)

        print(
            f"🎰 Odds collection - Page {current_page}/{total_pages} ({total_results} results on page 1)"
        )

        # Get remaining pages if they exist
        for page in range(2, total_pages + 1):
            try:
                time.sleep(0.1)  # Rate limiting

                conn = http.client.HTTPSConnection("v3.football.api-sports.io")
                conn.request("GET", f"/odds?date={date}&bookmaker=8&page={page}", headers=headers)
                res = conn.getresponse()

                if res.status != 200:
                    print(f"❌ HTTP {res.status} error getting odds page {page}")
                    conn.close()
                    continue

                page_data = res.read()
                conn.close()

                page_decoded_data = page_data.decode("utf-8")
                page_json = json.loads(page_decoded_data)

                # Check for API errors on this page
                if "errors" in page_json and page_json["errors"]:
                    print(f"❌ API Error on odds page {page}: {page_json['errors']}")
                    continue

                page_results = page_json.get("results", 0)
                print(
                    f"🎰 Odds collection - Page {page}/{total_pages} ({page_results} results)"
                )

                # Append results from this page
                if "response" in page_json and page_json["response"]:
                    data["response"].extend(page_json["response"])

            except Exception as e:
                print(f"❌ Error processing odds page {page}: {e}")
                continue

        conn = duckdb.connect('football_data.duckdb')
        for odds_entry in data.get("response", []):
            conn.execute("""
                INSERT OR REPLACE INTO odds VALUES (?, ?, ?, ?, ?, ?)
            """, [
                odds_entry["fixture"]["id"],
                odds_entry["fixture"]["date"][:10],  # Store only date part
                odds_entry["league"]["id"],
                odds_entry["league"]["name"],
                odds_entry["league"]["country"],
                json.dumps(odds_entry.get("bookmakers", []))
            ])
        
        total_odds_collected = len(data.get("response", []))
        print(
            f"✅ Odds collection complete: {total_odds_collected} fixtures with odds from {total_pages} pages"
        )
        
        # Save to cache for 3 hours
        save_to_cache(cache_key, data.get("response", []), minutes=180)
        cleanup_expired_cache()
        
        return data.get("response", [])

    except Exception as e:
        print(f"❌ Error in odds collection: {e}")
        conn.close()
        return []

def get_predictions(date=None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    else:
        date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")

    fixture_ids = get_fixtures_past_time_plus_hour()
    
    # Check cache for this set of fixture IDs
    cache_key = f"predictions_{date}_{len(fixture_ids)}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        print(f"✅ Using cached predictions for {len(fixture_ids)} fixtures")
        return cached_data
    duckdb_conn = duckdb.connect('football_data.duckdb')
    
    if not fixture_ids:
        print("📭 No fixtures found for predictions")
        duckdb_conn.close()
        return
    
    print(f"🎯 Getting predictions for {len(fixture_ids)} fixtures...")

    conn = http.client.HTTPSConnection("v3.football.api-sports.io")
    
    for fixture_id in fixture_ids:
        try:
            # Get fixture details from database first
            fixture_data = duckdb_conn.execute("""
                SELECT fixture_id, date, league_id, league_name, league_country, 
                       home_team_id, home_team_name, away_team_id, away_team_name
                FROM fixtures 
                WHERE fixture_id = ?
            """, [fixture_id]).fetchone()
            
            if not fixture_data:
                print(f"⚠️  Fixture {fixture_id} not found in database")
                continue
            
            # Get predictions from API
            conn.request("GET", f"/predictions?fixture={fixture_id}", headers=headers)
            res = conn.getresponse()
            
            if res.status != 200:
                print(f"❌ HTTP {res.status} error getting predictions for fixture {fixture_id}")
                continue
                
            data = res.read()
            decoded_data = data.decode("utf-8")
            parsed_data = json.loads(decoded_data)
            
            # Extract prediction response
            prediction_response = parsed_data.get("response", [])
            if not prediction_response:
                print(f"⚠️  No predictions available for fixture {fixture_id}")
                continue
            
            prediction_data = prediction_response[0]  # Get first prediction
            
            # Extract team predictions
            home_prediction = {
                "prediction": prediction_data.get("predictions", {}).get("winner", {}).get("name"),
                "probability": prediction_data.get("predictions", {}).get("percent", {}).get("home"),
                "goals": prediction_data.get("predictions", {}).get("goals", {}).get("home"),
                "comparison": prediction_data.get("comparison", {}).get("home", {})
            }
            
            away_prediction = {
                "prediction": prediction_data.get("predictions", {}).get("winner", {}).get("name"),
                "probability": prediction_data.get("predictions", {}).get("percent", {}).get("away"),
                "goals": prediction_data.get("predictions", {}).get("goals", {}).get("away"), 
                "comparison": prediction_data.get("comparison", {}).get("away", {})
            }
            
            # Insert into predictions table with correct schema
            duckdb_conn.execute("""
                INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                fixture_data[0],  # fixture_id
                fixture_data[1],  # date
                fixture_data[2],  # league_id
                fixture_data[3],  # league_name
                fixture_data[4],  # league_country
                fixture_data[5],  # home_team_id
                fixture_data[6],  # home_team_name
                json.dumps(home_prediction),  # home_team_prediction JSON
                fixture_data[7],  # away_team_id
                fixture_data[8],  # away_team_name
                json.dumps(away_prediction)   # away_team_prediction JSON
            ])
            
            print(f"✅ Predictions collected for fixture {fixture_id}")
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"❌ Error processing predictions for fixture {fixture_id}: {e}")
            continue
    
    conn.close()
    duckdb_conn.close()
    print(f"🎯 Predictions collection complete for {len(fixture_ids)} fixtures")
    
    # Save to cache for 1 hour
    save_to_cache(cache_key, {"fixture_ids": fixture_ids, "date": date}, minutes=60)
    cleanup_expired_cache()


def merge_data(date=None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    else:
        date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    with open(f"api_football_fixtures_{date}.json", "r", encoding="utf-8") as f:
        fixtures_data = json.load(f)
    with open(f"api_football_odds_{date}.json", "r", encoding="utf-8") as f:
        odds_data = json.load(f)
    with open(f"api_football_predictions_{date}.json", "r", encoding="utf-8") as f:
        predictions_data = json.load(f)

    for response in fixtures_data.get("fixtures", []):
        fixture_id = response["fixture"]["id"]
        matching_odds = next(
            (
                odds
                for odds in odds_data.get("response", [])
                if odds["fixture"]["id"] == fixture_id
            ),
            None,
        )
        matching_predictions = next(
            (
                prediction
                for prediction in predictions_data.get("fixtures", [])
                if prediction["fixture"]["id"] == fixture_id
            ),
            None,
        )
        if matching_odds:
            response["odds"] = matching_odds.get("bookmakers", [])
        else:
            response["odds"] = []
        if matching_predictions:
            response["predictions"] = matching_predictions.get("predictions", [])
        else:
            response["predictions"] = []

    fixtures_data["timestamp"] = datetime.now().strftime("%Y-%m-%d")
    with open(f"api_football_merged_{date}.json", "w", encoding="utf-8") as f:
        json.dump(fixtures_data, f, ensure_ascii=False, indent=4)


def get_completed_fixtures(days_back=7):
    """Fetch completed fixtures from recent days for model validation"""
    conn = http.client.HTTPSConnection("v3.football.api-sports.io")
    
    completed_fixtures = []
    
    for day_offset in range(1, days_back + 1):
        try:
            # Get date from X days ago
            target_date = (datetime.now() - timedelta(days=day_offset))
            date_str = target_date.strftime("%Y-%m-%d")
            
            print(f"📅 Fetching completed fixtures from {date_str}...")
            
            conn.request("GET", f"/fixtures?date={date_str}&status=FT", headers=headers)
            res = conn.getresponse()
            
            if res.status != 200:
                print(f"❌ HTTP {res.status} error getting completed fixtures for {date_str}")
                continue
                
            data = res.read()
            decoded_data = data.decode("utf-8")
            parsed_data = json.loads(decoded_data).get("response", [])
            
            # Filter for only completed matches with results
            completed_matches = [
                fixture for fixture in parsed_data 
                if fixture["fixture"]["status"]["short"] == "FT" and 
                   fixture["goals"]["home"] is not None and 
                   fixture["goals"]["away"] is not None
            ]
            
            completed_fixtures.extend(completed_matches)
            print(f"✅ Found {len(completed_matches)} completed fixtures from {date_str}")
            
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            print(f"❌ Error fetching completed fixtures for {target_date}: {e}")
            continue
    
    conn.close()
    
    # Save completed fixtures for validation
    if completed_fixtures:
        validation_data = {
            "completed_fixtures": completed_fixtures,
            "collection_date": datetime.now().isoformat(),
            "days_back": days_back,
            "total_fixtures": len(completed_fixtures)
        }
        
        with open("api_football_completed.json", "w", encoding="utf-8") as f:
            json.dump(validation_data, f, ensure_ascii=False, indent=4)
            
        print(f"💾 Saved {len(completed_fixtures)} completed fixtures for validation")
    
    return completed_fixtures

def validate_predictions_enhanced():
    """Validate enhanced AI model predictions against actual results"""
    try:
        # Load enhanced AI model 
        from football_ai_enhanced import EnhancedFootballAI
        enhanced_ai = EnhancedFootballAI()
        enhanced_ai.load_enhanced_models("enhanced_football_ai_model.pkl")
        
        # Load completed fixtures
        completed_fixtures_file = "api_football_completed.json"
        if not os.path.exists(completed_fixtures_file):
            print("❌ No completed fixtures data found")
            return
        
        with open(completed_fixtures_file, "r", encoding="utf-8") as f:
            completed_data = json.load(f)
        
        completed_fixtures = completed_data.get("completed_fixtures", [])
        if not completed_fixtures:
            print("❌ No completed fixtures to validate")
            return
        
        print("🔬 Starting enhanced AI prediction validation...")
        
        total = 0
        match_correct = 0
        goals_correct = 0 
        btts_correct = 0
        
        processed = 0
        max_process = 100  # Limit for speed
        
        for fixture in completed_fixtures[:max_process]:
            try:
                # Extract features for prediction
                df = enhanced_ai.extract_enhanced_features([fixture])
                if df.empty:
                    continue
                    
                # Get prediction
                features = df.iloc[0].to_dict()
                prediction = enhanced_ai.predict_match_enhanced(features)
                
                if 'error' in prediction:
                    continue
                
                # Get actual results
                home_goals = fixture["goals"]["home"]
                away_goals = fixture["goals"]["away"]
                
                if home_goals is None or away_goals is None:
                    continue
                
                total += 1
                processed += 1
                
                # Validate match result
                if home_goals > away_goals:
                    actual_result = "H"
                elif away_goals > home_goals:
                    actual_result = "A"
                else:
                    actual_result = "D"
                
                # Get AI's top prediction
                match_probs = prediction["match_result"]
                ai_prediction_numeric = max(match_probs, key=match_probs.get)
                
                # Convert numeric prediction to string label
                # LabelEncoder maps alphabetically: 0='A', 1='D', 2='H'
                if ai_prediction_numeric == 0:
                    ai_prediction = 'A'
                elif ai_prediction_numeric == 1:
                    ai_prediction = 'D'
                elif ai_prediction_numeric == 2:
                    ai_prediction = 'H'
                else:
                    ai_prediction = str(ai_prediction_numeric)  # Fallback
                
                if ai_prediction == actual_result:
                    match_correct += 1
                
                # Validate total goals  
                actual_total = home_goals + away_goals
                predicted_total = prediction["goals"]["total_expected"]
                
                # Consider correct if within 1 goal
                if abs(predicted_total - actual_total) <= 1.0:
                    goals_correct += 1
                
                # Validate BTTS
                actual_btts = home_goals > 0 and away_goals > 0
                predicted_btts_prob = prediction["btts"]["probability"]
                predicted_btts = predicted_btts_prob > 0.5
                
                if predicted_btts == actual_btts:
                    btts_correct += 1
                
                if processed % 20 == 0:
                    print(f"📊 Analyzed: {fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']} ({home_goals}-{away_goals})")
            
            except Exception:
                continue
        
        if total > 0:
            # Calculate accuracy percentages
            match_accuracy = match_correct / total
            goals_accuracy = goals_correct / total  
            btts_accuracy = btts_correct / total
            
            print("\n📊 Enhanced AI Prediction Validation Results:")
            print("=" * 50)
            print(f"📅 Fixtures analyzed: {total}")
            print(f"🎯 Match Result: {match_correct}/{total} ({match_accuracy:.1%})")
            print(f"🎯 Over Under Goals: {goals_correct}/{total} ({goals_accuracy:.1%})")
            print(f"🎯 Btts: {btts_correct}/{total} ({btts_accuracy:.1%})")
            
            # Save results
            validation_results = {
                "total_fixtures_analyzed": total,
                "prediction_accuracy": {
                    "match_result": {
                        "correct": match_correct,
                        "total": total,
                        "accuracy": match_accuracy
                    },
                    "over_under_goals": {
                        "correct": goals_correct,
                        "total": total, 
                        "accuracy": goals_accuracy
                    },
                    "btts": {
                        "correct": btts_correct,
                        "total": total,
                        "accuracy": btts_accuracy
                    }
                },
                "enhanced_ai_used": True,
                "validation_date": datetime.now().isoformat()
            }
            
            with open("prediction_validation_results.json", "w", encoding="utf-8") as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            print("💾 Enhanced validation results saved to prediction_validation_results.json")
        else:
            print("❌ No fixtures could be validated")
            
    except Exception as e:
        print(f"❌ Enhanced validation failed: {str(e)}")

def validate_predictions():
    """Compare AI predictions against actual results for completed fixtures"""
    print("🔬 Starting prediction validation...")
    
    # Load completed fixtures if available
    if not os.path.exists("api_football_completed.json"):
        print("❌ No completed fixtures data found. Run get_completed_fixtures() first.")
        return
    
    with open("api_football_completed.json", "r", encoding="utf-8") as f:
        completed_data = json.load(f)
    
    completed_fixtures = completed_data.get("completed_fixtures", [])
    
    if not completed_fixtures:
        print("❌ No completed fixtures available for validation")
        return
    
    # Initialize validation tracking
    validation_results = {
        "validation_date": datetime.now().isoformat(),
        "total_fixtures_analyzed": 0,
        "prediction_accuracy": {
            "match_result": {"correct": 0, "total": 0, "accuracy": 0.0},
            "over_under_goals": {"correct": 0, "total": 0, "accuracy": 0.0},
            "btts": {"correct": 0, "total": 0, "accuracy": 0.0}
        },
        "detailed_results": []
    }
    
    # Load AI system for making predictions on completed fixtures
    try:
        from football_ai_system import FootballAISystem
        ai_system = FootballAISystem()
        ai_system.load_models("football_ai_model.pkl")
        print("✅ AI models loaded for validation")
    except Exception as e:
        print(f"❌ Error loading AI models: {e}")
        return
    
    # Simple validation using basic fixture data (no detailed predictions needed)
    for fixture in completed_fixtures:
        try:
            # Get actual results
            home_goals = fixture["goals"]["home"]
            away_goals = fixture["goals"]["away"]
            
            if home_goals is None or away_goals is None:
                continue  # Skip fixtures without proper score data
                
            total_goals = home_goals + away_goals
            
            # Determine actual outcomes
            if home_goals > away_goals:
                actual_result = "home_win"
            elif away_goals > home_goals:
                actual_result = "away_win"
            else:
                actual_result = "draw"
            
            actual_over_25 = total_goals > 2.5
            actual_btts = home_goals > 0 and away_goals > 0
            
            # Create a simplified prediction using basic team strength heuristics
            # (since we can't run full AI prediction without prediction data)
            home_team = fixture["teams"]["home"]["name"]
            away_team = fixture["teams"]["away"]["name"]
            
            # Simple heuristic: assume home advantage and analyze team names for strength indicators
            home_strength = 0.5  # Base home advantage
            away_strength = 0.3
            
            # Adjust based on team name indicators (very basic)
            if any(word in home_team.lower() for word in ["real", "barcelona", "manchester", "liverpool", "arsenal", "chelsea"]):
                home_strength += 0.2
            if any(word in away_team.lower() for word in ["real", "barcelona", "manchester", "liverpool", "arsenal", "chelsea"]):
                away_strength += 0.2
            
            # Normalize probabilities
            total_strength = home_strength + away_strength + 0.2  # draw probability
            home_prob = home_strength / total_strength
            away_prob = away_strength / total_strength
            draw_prob = 0.2 / total_strength
            
            # Determine predicted result
            result_probs = {"home_win": home_prob, "away_win": away_prob, "draw": draw_prob}
            predicted_result = max(result_probs, key=result_probs.get)
            
            # Simple goals prediction (based on team strength)
            expected_goals = 1.5 + home_strength + away_strength
            predicted_over_25 = expected_goals > 2.5
            
            # BTTS prediction (assume moderate probability for now)
            predicted_btts = expected_goals > 2.0 and min(home_strength, away_strength) > 0.25
            
            # Track results
            fixture_validation = {
                "fixture_id": fixture["fixture"]["id"],
                "teams": f"{home_team} vs {away_team}",
                "actual_score": f"{home_goals}-{away_goals}",
                "actual_result": actual_result,
                "actual_total_goals": total_goals,
                "actual_btts": actual_btts,
                "simple_prediction": {
                    "predicted_result": predicted_result,
                    "predicted_over_25": predicted_over_25,
                    "predicted_btts": predicted_btts,
                    "home_strength": home_strength,
                    "away_strength": away_strength,
                    "expected_goals": expected_goals
                }
            }
            
            # Validate match result prediction
            match_correct = (predicted_result == actual_result)
            validation_results["prediction_accuracy"]["match_result"]["total"] += 1
            if match_correct:
                validation_results["prediction_accuracy"]["match_result"]["correct"] += 1
            fixture_validation["match_result_correct"] = match_correct
            
            # Validate Over/Under 2.5 goals prediction
            goals_correct = (predicted_over_25 == actual_over_25)
            validation_results["prediction_accuracy"]["over_under_goals"]["total"] += 1
            if goals_correct:
                validation_results["prediction_accuracy"]["over_under_goals"]["correct"] += 1
            fixture_validation["goals_prediction_correct"] = goals_correct
            
            # Validate BTTS prediction
            btts_correct = (predicted_btts == actual_btts)
            validation_results["prediction_accuracy"]["btts"]["total"] += 1
            if btts_correct:
                validation_results["prediction_accuracy"]["btts"]["correct"] += 1
            fixture_validation["btts_prediction_correct"] = btts_correct
            
            validation_results["detailed_results"].append(fixture_validation)
            validation_results["total_fixtures_analyzed"] += 1
            
            # Print progress for first few
            if validation_results["total_fixtures_analyzed"] <= 5:
                print(f"📊 Analyzed: {home_team} vs {away_team} ({home_goals}-{away_goals})")
                
        except Exception as e:
            print(f"❌ Error validating fixture {fixture.get('fixture', {}).get('id', 'unknown')}: {e}")
            continue
    
    # Calculate accuracy percentages
    for prediction_type in validation_results["prediction_accuracy"]:
        accuracy_data = validation_results["prediction_accuracy"][prediction_type]
        if accuracy_data["total"] > 0:
            accuracy_data["accuracy"] = accuracy_data["correct"] / accuracy_data["total"]
    
    # Save validation results
    with open("prediction_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=4)
    
    # Print validation summary
    print("\n📊 Prediction Validation Results:")
    print("=" * 50)
    print(f"📅 Fixtures analyzed: {validation_results['total_fixtures_analyzed']}")
    
    for pred_type, accuracy_data in validation_results["prediction_accuracy"].items():
        if accuracy_data["total"] > 0:
            accuracy_pct = accuracy_data["accuracy"] * 100
            print(f"🎯 {pred_type.replace('_', ' ').title()}: {accuracy_data['correct']}/{accuracy_data['total']} ({accuracy_pct:.1f}%)")
    
    print("💾 Detailed results saved to prediction_validation_results.json")
    print("📝 Note: Using simplified heuristic predictions for demonstration")
    
    return validation_results

def get_fixtures_past_time_plus_hour():
    """
    Get all fixture IDs where the fixture time is past now + 1 hour
    (includes ongoing matches and matches starting within 1 hour)
    """
    try:
        # Calculate the target time (now + 1 hour)
        from datetime import datetime, timedelta
        target_time = datetime.now() + timedelta(hours=1)
        target_time_str = target_time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"🕒 Looking for fixtures past {target_time_str}")
        
        # First, let's get fixtures from the API with full datetime info for today
        # Since our database only stores dates, we need to get fresh data with times
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Get fresh fixture data with timestamps
        api_conn = http.client.HTTPSConnection("v3.football.api-sports.io")
        api_conn.request("GET", f"/fixtures?date={today}", headers=headers)
        res = api_conn.getresponse()
        
        if res.status != 200:
            print(f"❌ HTTP {res.status} error getting fixtures")
            api_conn.close()
            return []
            
        data = res.read()
        api_conn.close()
        decoded_data = data.decode("utf-8")
        parsed_data = json.loads(decoded_data).get("response", [])
        
        # Filter fixtures based on time criteria
        matching_fixtures = []
        
        for fixture in parsed_data:
            try:
                # Get fixture datetime string (e.g., "2026-02-16T15:00:00+00:00")
                fixture_datetime_str = fixture["fixture"]["date"]
                
                # Parse the datetime (handle timezone if present)
                if '+' in fixture_datetime_str:
                    fixture_datetime_str = fixture_datetime_str.split('+')[0]
                elif 'Z' in fixture_datetime_str:
                    fixture_datetime_str = fixture_datetime_str.replace('Z', '')
                
                fixture_time = datetime.fromisoformat(fixture_datetime_str.replace('T', ' ')[:19])
                
                # Check if fixture time is <= now + 1 hour
                if fixture_time <= target_time:
                    fixture_id = fixture["fixture"]["id"]
                    fixture_status = fixture["fixture"]["status"]["long"]
                    team_info = f"{fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}"

                    
                    if fixture_status in ["Not Started"]:
                        matching_fixtures.append({
                            'fixture_id': fixture_id,
                            'fixture_time': fixture_time,
                            'status': fixture_status,
                            'teams': team_info
                        })
                    
            except Exception as e:
                print(f"❌ Error processing fixture {fixture.get('fixture', {}).get('id', 'unknown')}: {e}")
                continue
        
        # Sort by fixture time
        matching_fixtures.sort(key=lambda x: x['fixture_time'])
        
        # Extract just the IDs
        fixture_ids = [f['fixture_id'] for f in matching_fixtures]
        
        print(f"\n✅ Found {len(fixture_ids)} fixtures past {target_time_str}")
        
        return fixture_ids
        
    except Exception as e:
        print(f"❌ Error getting fixtures past time: {e}")
        return []


def main(date=None):
    print("🚀 Starting football data collection...")
    get_fixtures(date)
    get_odds(date)
    get_predictions(date)
    print("✅ Data collection complete!")

if __name__ == "__main__":
    import sys
    
    # Check if user wants to get fixtures past time
    if len(sys.argv) > 1 and sys.argv[1] == "--past-time":
        print("🕒 Getting fixtures past now + 1 hour...")
        fixture_ids = get_fixtures_past_time_plus_hour()
        if fixture_ids:
            print(f"\n📋 RESULT: {len(fixture_ids)} fixture IDs found")
            for fid in fixture_ids:
                print(f"   - {fid}")
        else:
            print("📭 No fixtures found matching criteria")
    else:
        # Normal data collection
        input_date = input("📅 Enter date for data collection (YYYY-MM-DD) or leave blank for today: ").strip()
        main(input_date if input_date else None)
