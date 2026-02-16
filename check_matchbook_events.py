import json

# Load Matchbook simplified events
with open('matchbook_football_events_simplified.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

events = data.get('events', [])
print(f"Total Matchbook events: {len(events)}")
print("\n" + "=" * 70)
print("First 30 Matchbook event names:")
print("=" * 70)

for i, event in enumerate(events[:30]):
    event_name = event.get('name', 'N/A')
    event_id = event.get('id', 'N/A')
    print(f"{i+1:3d}. {event_name} (ID: {event_id})")

# Look for Al Ittihad or Smouha specifically
print("\n" + "=" * 70)
print("Events containing 'Ittihad' or 'Smouha':")
print("=" * 70)

for event in events:
    event_name = event.get('name', '')
    if 'ittihad' in event_name.lower() or 'smouha' in event_name.lower():
        print(f"  - {event_name}")
