import requests
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

load_dotenv()

INTERVAL = 10


class matchbookExchange:
    def __init__(self) -> None:
        self.url = "https://api.matchbook.com/bpapi/rest/security/session"
        self.payload = {
            "username": os.getenv("MATCHBOOK_USERNAME"),
            "password": os.getenv("MATCHBOOK_PASSWORD"),
        }
        self.header = {
            "content-type": "application/json;charset=UTF-8",
            "accept": "*/*",
        }
        self.db = None
        self.token = None

    def login(self):
        try:
            r = requests.post(self.url, data=json.dumps(self.payload), headers=self.header)
            
            # Check HTTP status first
            if r.status_code != 200:
                print(f"❌ Matchbook login HTTP error: {r.status_code}")
                print(f"❌ Response text: {r.text[:200]}")
                raise Exception(f"Matchbook API returned HTTP {r.status_code}: {r.text[:100]}")
            
            # Check if response is valid JSON
            if not r.text.strip():
                print("❌ Matchbook login returned empty response")
                raise Exception("Matchbook API returned empty response")
            
            try:
                data = r.json()
            except json.JSONDecodeError as e:
                print(f"❌ Matchbook login invalid JSON response: {r.text[:200]}")
                raise Exception(f"Matchbook API returned invalid JSON: {e}")
            
            # Check for errors in response
            if 'errors' in data:
                error_msg = data.get('errors', [{}])[0].get('messages', ['Unknown error'])[0]
                raise Exception(f"Matchbook login failed: {error_msg}")
            
            if 'session-token' not in data:
                raise Exception("Matchbook login successful but no session token received")
                
        except requests.RequestException as e:
            print(f"❌ Matchbook login connection error: {e}")
            raise Exception(f"Failed to connect to Matchbook API: {e}")
        except Exception as e:
            print(f"❌ Matchbook login error: {e}")
            raise
            raise Exception(f"Matchbook login failed: No session-token in response. Response: {data}")
        
        self.token = data["session-token"]

    def get_account(self):
        url = "https://api.matchbook.com/edge/rest/account"
        headers = {
            "Accept": "application/json",
            "User-Agent": "api-doc-test-client",
            "session-token": self.token,
        }
        response = requests.request("GET", url, headers=headers)
        data = response.json()
        return data

    def get_current_bets(self):
        url = "https://api.matchbook.com/edge/rest/reports/v2/bets/current"
        headers = {
            "Accept": "application/json",
            "User-Agent": "api-doc-test-client",
            "session-token": self.token,
        }
        params = {
            "per-page": 1000,
        }
        response = requests.request("GET", url, headers=headers, params=params)
        data = response.json()
        return data

    def get_settled_bets(self, days=2):
        url = "https://api.matchbook.com/edge/rest/reports/v2/bets/settled"
        headers = {
            "Accept": "application/json",
            "User-Agent": "api-doc-test-client",
            "session-token": self.token,
        }
        #An ISO8601 timestamp e.g. 2017-01-01T12:00:00.000Z. Only bets settled after the timestamp will be returned.
        current_time = datetime.now()
        current_time_iso = current_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        time_minus_days = current_time - timedelta(days=days)
        time_minus_days_iso = time_minus_days.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        params = {
            "per-page": 1000,
            "before": current_time_iso,
            "after": time_minus_days_iso,
        }
        response = requests.request("GET", url, headers=headers, params=params)
        data = response.json()
        return data

    def get_sports(self):
        url = "https://api.matchbook.com/edge/rest/lookups/sports"
        headers = {
            "Accept": "application/json",
            "User-Agent": "api-doc-test-client",
            "session-token": self.token,
        }
        params = {"name": "asc", "status": "active", "per-page": 1000}
        r = requests.request("GET", url, headers=headers, params=params)
        data = r.json()
        return data

    def get_football_events(self, sports_id=15):
        url = f"https://api.matchbook.com/edge/rest/events?sport-ids={sports_id}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "api-doc-test-client",
            "session-token": self.token,
        }
        current_time = datetime.now()
        current_time_unix = int(current_time.timestamp())
        time_plus_1_hours = current_time + timedelta(hours=1)
        unix_timestamp_plus_1_hours = int(time_plus_1_hours.timestamp())
        params = {
            "per-page": 1000,
            "states": "open",
            "after": str(current_time_unix),
            "before": str(unix_timestamp_plus_1_hours),
            "include-prices": True,
            "side": "back",
        }
        response = requests.request("GET", url, headers=headers, params=params)
        data = response.json()
        return data

    def get_markets(self, event_id):
        url = f"https://api.matchbook.com/edge/rest/events/{event_id}/markets"

        headers = {
            "accept": "application/json",
            "User-Agent": "api-doc-test-client",
            "session-token": self.token,
        }
        params = {"per-page": 1000}
        response = requests.get(url, headers=headers, params=params)

        data = response.json()
        return data

    def get_runners(self, event_id, market_id):
        url = f"https://api.matchbook.com/edge/rest/events/{event_id}/markets/{market_id}/runners"

        headers = {
            "accept": "application/json",
            "User-Agent": "api-doc-test-client",
            "session-token": self.token,
        }

        response = requests.get(url, headers=headers)

        data = response.json()
        return data

    def get_runner(self, event_id, market_id, runner_id):
        url = f"https://api.matchbook.com/edge/rest/events/{event_id}/markets/{market_id}/runners/{runner_id}"

        headers = {
            "accept": "application/json",
            "User-Agent": "api-doc-test-client",
            "session-token": self.token,
        }

        response = requests.get(url, headers=headers)

        data = response.json()
        return data

    def place_order(self, runner):
        url = "https://api.matchbook.com/edge/rest/v2/offers"

        payload = {
            "odds-type": "DECIMAL",
            "exchange-type": "back-lay",
            "offers": [
                {
                    "runner-id": runner.get("runner_id"),
                    "side": runner.get("side"),
                    "odds": runner.get("odds"),
                    "stake": runner.get("stake"),
                    "keep-in-play": False,
                }
            ],
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "api-doc-test-client",
            "Content-Type": "application/json",
            "session-token": self.token,
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        return response.json()

    def get_offer(self, offer_id):
        url = "https://api.matchbook.com/edge/rest/v2/offers"

        headers = {
            "Accept": "application/json",
            "User-Agent": "api-doc-test-client",
            "session-token": self.token,
        }

        params = {"offer_id": offer_id}

        response = requests.request("GET", url, headers=headers, params=params)
        return response.json()

    def cancel_all_orders(self, event_id):
        url = "https://api.matchbook.com/edge/rest/v2/offers"

        headers = {
            "Accept": "application/json",
            "User-Agent": "api-doc-test-client",
            "session-token": self.token,
        }

        params = {"event-ids": event_id}

        response = requests.request("DELETE", url, headers=headers, params=params)
        return response.json()

    def cancel_single_order(self, offer_id):
        url = f"https://api.matchbook.com/edge/rest/v2/offers/{offer_id}"

        headers = {
            "Accept": "application/json",
            "User-Agent": "api-doc-test-client",
            "session-token": self.token,
        }

        response = requests.request("DELETE", url, headers=headers)
        print(response.json())
        {"errors": [{"messages": ["No offer exists with id 123."]}]}
        if response.json().get("errors", {}):
            return True, 0
        else:
            if response.json().get("stake", 0) > response.json().get("remaining", 0):
                return True, response.json().get("stake", 0) - response.json().get(
                    "remaining", 0
                )
            else:
                return False, response.json().get("stake", 0)
        # return response.json()


def process_markets(markets):
    """Extract simplified market data with only essential information"""
    simplified_markets = []

    for market in markets:
        market_info = {
            "market_name": market.get("name"),
            "market_id": market.get("id"),
            "runners": [],
        }

        # Process each runner in the market
        for runner in market.get("runners", []):
            runner_name = runner.get("name")

            # Find best back price (highest odds on back side)
            best_back_price = None
            best_back_amount = 0

            for price in runner.get("prices", []):
                if price.get("side") == "back":
                    odds = price.get("odds", 0)
                    amount = price.get("available-amount", 0)

                    # Keep the highest back odds
                    if best_back_price is None or odds > best_back_price:
                        best_back_price = odds
                        best_back_amount = amount

            if best_back_price is not None:
                market_info["runners"].append(
                    {
                        "runner_name": runner_name,
                        "runner_id": runner.get(
                            "id"
                        ),  # Include runner ID for bet placement
                        "best_back_odds": best_back_price,
                        "available_amount": best_back_amount,
                    }
                )

        # Only add markets that have runners with prices
        if market_info["runners"]:
            simplified_markets.append(market_info)

    return simplified_markets


# matchbook = matchbookExchange()
# matchbook.login()
# football_events = matchbook.get_football_events()
# football_dict = {}

# for event in football_events.get("events", []):
#     simplified_markets = process_markets(event.get("markets", []))

#     football_dict[event["name"]] = {
#         "start": event["start"],
#         "event_id": event.get("id"),
#         "simplified_markets": simplified_markets,
#         "total_markets": len(simplified_markets),
#     }

# with open("matchbook_football_events_simplified.json", "w", encoding="utf-8") as f:
#     json.dump(football_dict, f, ensure_ascii=False, indent=4)

# # Print summary
# print("📊 Matchbook Data Processing Summary")
# print("=" * 50)
# total_events = len(football_dict)
# total_markets = sum(event["total_markets"] for event in football_dict.values())

# print(f"📅 Events processed: {total_events}")
# print(f"📈 Total markets: {total_markets}")

# # Show sample of processed data
# for i, (event_name, event_data) in enumerate(football_dict.items()):
#     if i < 2:  # Show first 2 events
#         print(f"\n🏆 {event_name}")
#         print(f"   ⏰ Start: {event_data['start']}")
#         print(f"   📊 Markets: {event_data['total_markets']}")

#         # Show first few markets
#         for market in event_data["simplified_markets"][:3]:
#             print(f"   📈 {market['market_name']}:")
#             for runner in market["runners"][:2]:  # Show first 2 runners
#                 print(
#                     f"      • {runner['runner_name']}: {runner['best_back_odds']} (£{runner['available_amount']:.2f})"
#                 )