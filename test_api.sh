#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Base URL (change for production)
BASE_URL="${BASE_URL:-http://localhost:8080}"

echo -e "${BLUE}=== Quantitative Betting API Test Script ===${NC}\n"

# Health check
echo -e "${BLUE}1. Checking health...${NC}"
curl -s "$BASE_URL/health" | python -m json.tool
echo -e "\n"

# Get status
echo -e "${BLUE}2. Getting system status...${NC}"
curl -s "$BASE_URL/status" | python -m json.tool
echo -e "\n"

# Trigger workflow (dry run)
echo -e "${BLUE}3. Triggering workflow (dry run - auto_place_bets: false)...${NC}"
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"min_ev_threshold":0.08,"min_confidence":0.65,"max_daily_stake":5.0,"auto_place_bets":false}' \
  "$BASE_URL/run-workflow" | python -m json.tool
echo -e "\n"

# Wait a bit
echo -e "${BLUE}4. Waiting 5 seconds...${NC}"
sleep 5

# Check workflow status
echo -e "${BLUE}5. Checking workflow status...${NC}"
curl -s "$BASE_URL/workflow-status" | python -m json.tool
echo -e "\n"

# Get bet history
echo -e "${BLUE}6. Getting bet history (last 7 days)...${NC}"
curl -s "$BASE_URL/bet-history?days=7&limit=10" | python -m json.tool
echo -e "\n"

echo -e "${GREEN}=== Test complete ===${NC}"
