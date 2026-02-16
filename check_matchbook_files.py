import json
import os

print("Checking Matchbook data files...")
print("=" * 70)

files = [
    'matchbook_football_events.json',
    'matchbook_football_events_simplified.json'
]

for filename in files:
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            events = data.get('events', [])
            print(f"\n{filename}:")
            print(f"  Size: {size:,} bytes")
            print(f"  Events: {len(events)}")
            if events:
                print(f"  First event: {events[0].get('name', 'N/A')}")
    else:
        print(f"\n{filename}: NOT FOUND")
