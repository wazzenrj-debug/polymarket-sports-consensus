"""
Polymarket Sports Consensus Bot with Analytics

Pulls top 100 traders by profit (30 days), analyzes their sports positions,
tracks prediction accuracy, and generates HTML dashboards.

Requirements:
    pip install httpx pydantic

Usage:
    python polymarket_sports_consensus.py
    
Output:
    - sports_consensus_dashboard.html (current predictions)
    - analytics.html (historical accuracy)
    - predictions_history.json (data store)
"""

import httpx
import json
import os
import webbrowser
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
import time


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "leaderboard_window": "30d",      # 30 day profit leaderboard
    "leaderboard_limit": 200,          # Top 200 traders
    "min_position_size": 100,          # Minimum $100 position
    "min_traders_per_match": 5,        # Minimum traders to show a match
    "days_ahead": 7,                   # Look 7 days ahead
    "history_days": 90,                # Keep 90 days of history
    "output_file": "sports_consensus_dashboard.html",
    "analytics_file": "analytics.html",
    "history_file": "predictions_history.json",
    "rate_limit_delay": 0.1,           # Delay between API calls (seconds)
}

# API Endpoints
ENDPOINTS = {
    "leaderboard": "https://lb-api.polymarket.com/profit",
    "positions": "https://data-api.polymarket.com/positions",
    "gamma_events": "https://gamma-api.polymarket.com/events",
    "gamma_markets": "https://gamma-api.polymarket.com/markets",
    "gamma_sports": "https://gamma-api.polymarket.com/sports",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Trader:
    address: str
    username: str
    profit: float
    volume: float
    rank: int


@dataclass
class Position:
    trader_address: str
    condition_id: str
    token_id: str
    outcome: str
    outcome_index: int
    size: float  # Number of shares
    current_value: float  # USD value
    title: str
    slug: str
    event_slug: str
    end_date: Optional[str]
    icon: Optional[str]


@dataclass
class MatchOutcome:
    outcome_name: str
    outcome_index: int
    token_id: str
    trader_count: int = 0
    total_value: float = 0.0
    traders: list = field(default_factory=list)


@dataclass
class Match:
    condition_id: str
    title: str
    slug: str
    event_slug: str
    end_date: Optional[str]
    icon: Optional[str]
    sport: Optional[str] = None
    outcomes: dict = field(default_factory=dict)  # outcome_name -> MatchOutcome
    
    @property
    def total_traders(self) -> int:
        return sum(o.trader_count for o in self.outcomes.values())
    
    @property
    def total_volume(self) -> float:
        return sum(o.total_value for o in self.outcomes.values())
    
    @property
    def polymarket_url(self) -> str:
        return f"https://polymarket.com/event/{self.event_slug}"
    
    def consensus_strength(self) -> float:
        """Returns the highest percentage any single outcome has."""
        if self.total_traders == 0:
            return 0
        return max(o.trader_count / self.total_traders * 100 for o in self.outcomes.values())


@dataclass
class PredictionRecord:
    """Historical prediction record"""
    condition_id: str
    title: str
    event_slug: str
    end_date: str
    recorded_date: str
    
    # Consensus data
    trader_consensus_outcome: str
    trader_consensus_count: int
    trader_consensus_pct: float
    
    volume_consensus_outcome: str
    volume_consensus_amount: float
    volume_consensus_pct: float
    
    # All outcomes for reference
    outcomes_data: dict  # outcome_name -> {traders, volume, pct}
    
    # Resolution data (filled in later)
    resolved: bool = False
    resolved_date: Optional[str] = None
    winning_outcome: Optional[str] = None
    trader_consensus_correct: Optional[bool] = None
    volume_consensus_correct: Optional[bool] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


# =============================================================================
# Prediction History Manager
# =============================================================================

class PredictionHistoryManager:
    """Manages prediction history in JSON file"""
    
    def __init__(self, filepath: str = "predictions_history.json"):
        self.filepath = Path(filepath)
        self.predictions: list[PredictionRecord] = []
        self.load()
    
    def load(self):
        """Load predictions from JSON file"""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.predictions = [
                        PredictionRecord(**p) for p in data.get('predictions', [])
                    ]
                print(f"📂 Loaded {len(self.predictions)} historical predictions")
            except Exception as e:
                print(f"⚠️ Error loading history: {e}")
                self.predictions = []
        else:
            print("📂 No history file found, starting fresh")
            self.predictions = []
    
    def save(self):
        """Save predictions to JSON file"""
        data = {
            'last_updated': datetime.utcnow().isoformat(),
            'predictions': [p.to_dict() for p in self.predictions]
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved {len(self.predictions)} predictions to {self.filepath}")
    
    def add_prediction(self, match: Match):
        """Add a new prediction from a Match"""
        # Check if we already have this prediction
        existing = self.get_prediction(match.condition_id)
        if existing and not existing.resolved:
            # Update existing unresolved prediction
            existing = self._create_prediction_from_match(match)
            return
        
        if not existing:
            # Add new prediction
            prediction = self._create_prediction_from_match(match)
            self.predictions.append(prediction)
    
    def _create_prediction_from_match(self, match: Match) -> PredictionRecord:
        """Create PredictionRecord from Match"""
        # Find trader consensus (most traders)
        sorted_by_traders = sorted(
            match.outcomes.values(),
            key=lambda o: o.trader_count,
            reverse=True
        )
        trader_winner = sorted_by_traders[0]
        
        # Find volume consensus (most $$$)
        sorted_by_volume = sorted(
            match.outcomes.values(),
            key=lambda o: o.total_value,
            reverse=True
        )
        volume_winner = sorted_by_volume[0]
        
        # Build outcomes data
        outcomes_data = {}
        for outcome in match.outcomes.values():
            outcomes_data[outcome.outcome_name] = {
                'traders': outcome.trader_count,
                'volume': outcome.total_value,
                'trader_pct': (outcome.trader_count / match.total_traders * 100) if match.total_traders > 0 else 0,
                'volume_pct': (outcome.total_value / match.total_volume * 100) if match.total_volume > 0 else 0,
            }
        
        return PredictionRecord(
            condition_id=match.condition_id,
            title=match.title,
            event_slug=match.event_slug,
            end_date=match.end_date or "",
            recorded_date=datetime.utcnow().isoformat(),
            trader_consensus_outcome=trader_winner.outcome_name,
            trader_consensus_count=trader_winner.trader_count,
            trader_consensus_pct=(trader_winner.trader_count / match.total_traders * 100) if match.total_traders > 0 else 0,
            volume_consensus_outcome=volume_winner.outcome_name,
            volume_consensus_amount=volume_winner.total_value,
            volume_consensus_pct=(volume_winner.total_value / match.total_volume * 100) if match.total_volume > 0 else 0,
            outcomes_data=outcomes_data,
        )
    
    def get_prediction(self, condition_id: str) -> Optional[PredictionRecord]:
        """Get prediction by condition_id"""
        for p in self.predictions:
            if p.condition_id == condition_id:
                return p
        return None
    
    def cleanup_old_predictions(self, days: int = 90):
        """Remove predictions older than specified days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        original_count = len(self.predictions)
        
        self.predictions = [
            p for p in self.predictions
            if datetime.fromisoformat(p.recorded_date.replace("Z", "+00:00")).replace(tzinfo=None) > cutoff
        ]
        
        removed = original_count - len(self.predictions)
        if removed > 0:
            print(f"🧹 Cleaned up {removed} old predictions (>{days} days)")
    
    def get_stats(self) -> dict:
        """Calculate accuracy statistics"""
        resolved = [p for p in self.predictions if p.resolved]
        
        if not resolved:
            return {
                'total_predictions': len(self.predictions),
                'resolved': 0,
                'pending': len(self.predictions),
                'trader_accuracy': 0,
                'volume_accuracy': 0,
                'both_correct': 0,
                'neither_correct': 0,
            }
        
        trader_correct = sum(1 for p in resolved if p.trader_consensus_correct)
        volume_correct = sum(1 for p in resolved if p.volume_consensus_correct)
        both_correct = sum(1 for p in resolved if p.trader_consensus_correct and p.volume_consensus_correct)
        neither_correct = sum(1 for p in resolved if not p.trader_consensus_correct and not p.volume_consensus_correct)
        
        return {
            'total_predictions': len(self.predictions),
            'resolved': len(resolved),
            'pending': len(self.predictions) - len(resolved),
            'trader_accuracy': (trader_correct / len(resolved) * 100) if resolved else 0,
            'volume_accuracy': (volume_correct / len(resolved) * 100) if resolved else 0,
            'both_correct': both_correct,
            'neither_correct': neither_correct,
            'trader_wins': trader_correct,
            'volume_wins': volume_correct,
        }


# =============================================================================
# API Client
# =============================================================================

class PolymarketClient:
    def __init__(self):
        self.client = httpx.Client(
            http2=True,
            timeout=30.0,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        self.sports_tags = {}  # tag_id -> sport_name
    
    def _get(self, url: str, params: dict = None) -> dict:
        """Make GET request with rate limiting."""
        time.sleep(CONFIG["rate_limit_delay"])
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def check_market_resolution(self, condition_id: str) -> Optional[dict]:
        """Check if a market has resolved and get the winning outcome"""
        try:
            # Try to get market data from gamma API
            params = {"condition_id": condition_id}
            markets = self._get(ENDPOINTS["gamma_markets"], params)
            
            if not markets:
                return None
            
            market = markets[0] if isinstance(markets, list) else markets
            
            # Check if market is closed/resolved
            closed = market.get("closed", False)
            
            if not closed:
                return None
            
            # Find winning outcome (the one that resolved to YES)
            # In Polymarket, resolved markets have prices of 1.00 for winners
            outcomes = market.get("outcomes", [])
            tokens = market.get("tokens", [])
            
            # Method 1: Check prices (winners have price = 1.00)
            for i, token in enumerate(tokens):
                price = float(token.get("price", 0))
                if price >= 0.99:  # Essentially 1.00 (allowing for rounding)
                    outcome_name = outcomes[i] if i < len(outcomes) else None
                    return {
                        'resolved': True,
                        'winning_outcome': outcome_name,
                        'winning_token_id': token.get('token_id'),
                        'resolved_date': market.get('endDate') or datetime.utcnow().isoformat(),
                    }
            
            # Method 2: Check if there's a resolvedBy field
            if market.get('resolvedBy'):
                # Market is resolved but we need to find winner another way
                # Check acceptingOrders - if false and one outcome has high volume, it likely won
                pass
            
            return {
                'resolved': True,
                'winning_outcome': None,  # Couldn't determine winner
                'resolved_date': market.get('endDate') or datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            print(f"   ⚠️ Error checking resolution for {condition_id}: {e}")
            return None
    
    def get_sports_metadata(self) -> dict:
        """Fetch sports metadata to identify sports markets."""
        print("📊 Fetching sports metadata...")
        try:
            data = self._get(ENDPOINTS["gamma_sports"])
            for sport in data:
                if "tags" in sport and sport["tags"]:
                    # Tags might be comma-separated
                    for tag_id in str(sport["tags"]).split(","):
                        tag_id = tag_id.strip()
                        if tag_id.isdigit():
                            self.sports_tags[int(tag_id)] = sport.get("sport", "Sports")
            print(f"   Found {len(self.sports_tags)} sports tags")
            return self.sports_tags
        except Exception as e:
            print(f"   ⚠️ Could not fetch sports metadata: {e}")
            return {}
    
    def get_top_traders(self, window: str = "30d", limit: int = 100) -> list[Trader]:
        """Fetch top traders from leaderboard."""
        print(f"🏆 Fetching top {limit} traders by profit ({window})...")
        
        traders = []
        offset = 0
        batch_size = 50  # API might cap at 50 per request
        
        while len(traders) < limit:
            params = {
                "window": window,
                "limit": min(batch_size, limit - len(traders)),
                "offset": offset,
            }
            
            data = self._get(ENDPOINTS["leaderboard"], params)
            
            if not data:
                break
            
            for i, item in enumerate(data):
                trader = Trader(
                    address=item.get("proxyWallet", item.get("address", "")),
                    username=item.get("name") or item.get("pseudonym") or f"Trader_{len(traders)+1}",
                    profit=float(item.get("pnl", item.get("profit", 0))),
                    volume=float(item.get("volume", 0)),
                    rank=len(traders) + 1,
                )
                traders.append(trader)
                
                if len(traders) >= limit:
                    break
            
            # If we got fewer than requested, we've hit the end
            if len(data) < batch_size:
                break
                
            offset += batch_size
            print(f"   Fetched {len(traders)} traders so far...")
        
        print(f"   Found {len(traders)} traders total")
        if traders:
            print(f"   Top trader: {traders[0].username} (${traders[0].profit:,.2f} profit)")
        
        return traders
    
    def get_trader_positions(self, trader_address: str, min_size: float = 100) -> list[Position]:
        """Fetch positions for a single trader."""
        params = {
            "user": trader_address,
            "sizeThreshold": 1,  # Get all, filter later by value
            "limit": 500,
            "sortBy": "CURRENT",
            "sortDirection": "DESC",
        }
        
        try:
            data = self._get(ENDPOINTS["positions"], params)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return []
            raise
        
        positions = []
        for item in data:
            current_value = float(item.get("currentValue", 0))
            
            # Filter by minimum position size
            if current_value < min_size:
                continue
            
            position = Position(
                trader_address=trader_address,
                condition_id=item.get("conditionId", ""),
                token_id=item.get("asset", ""),
                outcome=item.get("outcome", ""),
                outcome_index=int(item.get("outcomeIndex", 0)),
                size=float(item.get("size", 0)),
                current_value=current_value,
                title=item.get("title", ""),
                slug=item.get("slug", ""),
                event_slug=item.get("eventSlug", ""),
                end_date=item.get("endDate"),
                icon=item.get("icon"),
            )
            positions.append(position)
        
        return positions
    
    def get_sports_events(self, days_ahead: int = 7) -> set[str]:
        """Fetch sports event slugs to filter positions."""
        print(f"🏈 Fetching sports events (next {days_ahead} days)...")
        
        sports_condition_ids = set()
        
        # Calculate date range
        now = datetime.utcnow()
        end_date = now + timedelta(days=days_ahead)
        
        # Fetch events with sports tag
        # We'll fetch multiple pages to get comprehensive data
        offset = 0
        limit = 100
        
        while True:
            params = {
                "closed": "false",
                "limit": limit,
                "offset": offset,
                "order": "end_date_min",
                "ascending": "true",
            }
            
            try:
                data = self._get(ENDPOINTS["gamma_events"], params)
            except Exception as e:
                print(f"   ⚠️ Error fetching events: {e}")
                break
            
            if not data:
                break
            
            for event in data:
                # Check if this is a sports event
                tags = event.get("tags", [])
                is_sports = False
                sport_name = None
                
                # Check tags
                if isinstance(tags, list):
                    for tag in tags:
                        tag_id = tag.get("id") if isinstance(tag, dict) else tag
                        if tag_id in self.sports_tags:
                            is_sports = True
                            sport_name = self.sports_tags[tag_id]
                            break
                        # Also check tag labels
                        tag_label = tag.get("label", "").lower() if isinstance(tag, dict) else ""
                        if any(s in tag_label for s in ["sports", "nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "hockey", "ufc", "mma", "tennis", "golf"]):
                            is_sports = True
                            sport_name = tag_label.title()
                            break
                
                # Also check category/slug for sports keywords - STRICT MATCHING
                slug = event.get("slug", "").lower()
                title = event.get("title", "").lower()
                
                # Explicit sports keywords - must match one of these
                sports_keywords = [
                    # Major leagues
                    "nfl", "nba", "mlb", "nhl", "mls", "wnba",
                    # Sports names
                    "football", "basketball", "baseball", "hockey", "soccer", 
                    "tennis", "golf", "boxing", "ufc", "mma", "cricket", "rugby",
                    "f1", "formula 1", "nascar", "racing",
                    # Match indicators
                    " vs ", " vs. ", " v ", 
                    # Teams (sample of major teams)
                    "lakers", "celtics", "warriors", "bulls", "heat", "knicks",
                    "chiefs", "eagles", "cowboys", "patriots", "49ers", "bills",
                    "yankees", "dodgers", "red sox", "cubs", "mets",
                    "cavaliers", "nets", "bucks", "suns", "nuggets", "thunder",
                    # Competitions
                    "super bowl", "world series", "stanley cup", "nba finals",
                    "playoff", "playoffs", "championship", "premier league", 
                    "champions league", "la liga", "serie a", "bundesliga",
                    "world cup", "euro 2024", "euros",
                    # Player props
                    "points", "rebounds", "assists", "touchdowns", "yards",
                    "home runs", "strikeouts", "goals", "saves",
                ]
                
                # Exclusion keywords - NOT sports even if they have "vs"
                exclude_keywords = [
                    "trump", "biden", "election", "president", "congress", "senate",
                    "bitcoin", "ethereum", "crypto", "btc", "eth", "price",
                    "fed", "inflation", "recession", "economy", "gdp",
                    "movie", "oscar", "grammy", "emmy", "album", "song",
                    "twitter", "tiktok", "youtube", "subscriber", "follower",
                    "elon", "musk", "zuckerberg", "bezos",
                    "ai", "openai", "chatgpt", "claude", "gpt",
                    "ukraine", "russia", "china", "war", "military",
                    "covid", "vaccine", "virus",
                    "temperature", "weather", "climate",
                    "spacex", "rocket", "mars", "moon",
                ]
                
                # Check for exclusions first
                has_exclusion = any(kw in slug or kw in title for kw in exclude_keywords)
                if has_exclusion:
                    continue
                
                # Must have at least one sports keyword
                has_sports_keyword = any(kw in slug or kw in title for kw in sports_keywords)
                
                if has_sports_keyword:
                    is_sports = True
                    if not sport_name:
                        sport_name = "Sports"
                
                if not is_sports:
                    continue
                
                # Check end date is within range
                end_date_str = event.get("endDate") or event.get("end_date_min")
                if end_date_str:
                    try:
                        event_end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                        if event_end.replace(tzinfo=None) > end_date:
                            continue
                        if event_end.replace(tzinfo=None) < now:
                            continue  # Already ended
                    except:
                        pass
                
                # Get markets/conditions for this event
                markets = event.get("markets", [])
                for market in markets:
                    condition_id = market.get("conditionId") or market.get("condition_id")
                    if condition_id:
                        sports_condition_ids.add(condition_id)
            
            offset += limit
            if len(data) < limit:
                break
            
            # Safety limit
            if offset > 1000:
                break
        
        print(f"   Found {len(sports_condition_ids)} sports markets")
        return sports_condition_ids
    
    def close(self):
        self.client.close()


# =============================================================================
# Data Aggregation
# =============================================================================

def aggregate_positions(
    traders: list[Trader],
    client: PolymarketClient,
    sports_condition_ids: set[str],
    min_position_size: float = 100,
    days_ahead: int = 7,
) -> dict[str, Match]:
    """Aggregate all trader positions into match consensus data."""
    
    print(f"\n📈 Analyzing positions for {len(traders)} traders...")
    print(f"   Minimum position size: ${min_position_size}")
    print(f"   Minimum traders per match: {CONFIG.get('min_traders_per_match', 5)}")
    print(f"   Looking ahead: {days_ahead} days")
    
    matches: dict[str, Match] = {}  # condition_id -> Match
    
    now = datetime.utcnow()
    end_date_limit = now + timedelta(days=days_ahead)
    
    for i, trader in enumerate(traders):
        print(f"\r   Processing trader {i+1}/{len(traders)}: {trader.username[:20]:<20}", end="", flush=True)
        
        try:
            positions = client.get_trader_positions(trader.address, min_size=min_position_size)
        except Exception as e:
            print(f"\n   ⚠️ Error fetching positions for {trader.username}: {e}")
            continue
        
        for pos in positions:
            # Filter: only sports markets
            if sports_condition_ids and pos.condition_id not in sports_condition_ids:
                # Double-check by title/slug with STRICT sports matching
                title_lower = pos.title.lower()
                slug_lower = pos.slug.lower()
                
                # Explicit sports keywords
                sports_keywords = [
                    "nfl", "nba", "mlb", "nhl", "mls", "wnba",
                    "football", "basketball", "baseball", "hockey", "soccer", 
                    "tennis", "golf", "boxing", "ufc", "mma", "cricket", "rugby",
                    "f1", "formula 1", "nascar", " vs ", " vs. ", " v ",
                    "lakers", "celtics", "warriors", "bulls", "heat", "knicks",
                    "chiefs", "eagles", "cowboys", "patriots", "49ers", "bills",
                    "yankees", "dodgers", "cavaliers", "nets", "bucks", "suns",
                    "super bowl", "world series", "stanley cup", "nba finals",
                    "playoff", "playoffs", "championship", "premier league",
                    "points", "rebounds", "assists", "touchdowns", "yards",
                    "home runs", "strikeouts", "goals", "saves",
                ]
                
                # Exclusion keywords
                exclude_keywords = [
                    "trump", "biden", "election", "president", "congress",
                    "bitcoin", "ethereum", "crypto", "btc", "eth", "price",
                    "fed", "inflation", "recession", "economy",
                    "movie", "oscar", "grammy", "emmy", "album",
                    "twitter", "tiktok", "youtube", "elon", "musk",
                    "ai", "openai", "chatgpt", "ukraine", "russia", "war",
                    "covid", "vaccine", "temperature", "weather", "spacex",
                ]
                
                # Check exclusions first
                has_exclusion = any(kw in title_lower or kw in slug_lower for kw in exclude_keywords)
                if has_exclusion:
                    continue
                
                # Must have sports keyword
                has_sports = any(kw in title_lower or kw in slug_lower for kw in sports_keywords)
                if not has_sports:
                    continue
            
            # Filter: check end date is in future and within range
            if pos.end_date:
                try:
                    pos_end = datetime.fromisoformat(pos.end_date.replace("Z", "+00:00"))
                    if pos_end.replace(tzinfo=None) < now:
                        continue  # Already ended
                    if pos_end.replace(tzinfo=None) > end_date_limit:
                        continue  # Too far in future
                except:
                    pass
            
            # Get or create match
            if pos.condition_id not in matches:
                matches[pos.condition_id] = Match(
                    condition_id=pos.condition_id,
                    title=pos.title,
                    slug=pos.slug,
                    event_slug=pos.event_slug,
                    end_date=pos.end_date,
                    icon=pos.icon,
                )
            
            match = matches[pos.condition_id]
            
            # Get or create outcome
            if pos.outcome not in match.outcomes:
                match.outcomes[pos.outcome] = MatchOutcome(
                    outcome_name=pos.outcome,
                    outcome_index=pos.outcome_index,
                    token_id=pos.token_id,
                )
            
            outcome = match.outcomes[pos.outcome]
            outcome.trader_count += 1
            outcome.total_value += pos.current_value
            outcome.traders.append({
                "address": trader.address,
                "username": trader.username,
                "rank": trader.rank,
                "value": pos.current_value,
            })
    
    print(f"\n   Found {len(matches)} matches with positions from top traders")
    
    return matches


def update_resolved_predictions(history: PredictionHistoryManager, client: PolymarketClient):
    """Check unresolved predictions for resolutions"""
    print("\n🔍 Checking for resolved markets...")
    
    unresolved = [p for p in history.predictions if not p.resolved]
    
    if not unresolved:
        print("   No unresolved predictions to check")
        return
    
    print(f"   Checking {len(unresolved)} unresolved predictions...")
    
    resolved_count = 0
    for prediction in unresolved:
        result = client.check_market_resolution(prediction.condition_id)
        
        if result and result.get('resolved'):
            winning_outcome = result.get('winning_outcome')
            
            if winning_outcome:
                prediction.resolved = True
                prediction.resolved_date = result.get('resolved_date')
                prediction.winning_outcome = winning_outcome
                prediction.trader_consensus_correct = (winning_outcome == prediction.trader_consensus_outcome)
                prediction.volume_consensus_correct = (winning_outcome == prediction.volume_consensus_outcome)
                resolved_count += 1
                
                # Show result
                trader_status = "✅" if prediction.trader_consensus_correct else "❌"
                volume_status = "✅" if prediction.volume_consensus_correct else "❌"
                print(f"   {trader_status} Traders | {volume_status} Volume | {prediction.title[:50]}")
    
    if resolved_count > 0:
        print(f"   ✅ Updated {resolved_count} newly resolved predictions")
    else:
        print(f"   No new resolutions found")


# =============================================================================
# HTML Dashboard Generator
# =============================================================================

def generate_html_dashboard(matches: dict[str, Match], traders: list[Trader]) -> str:
    """Generate a beautiful HTML dashboard."""
    
    # Filter out matches with less than minimum traders
    min_traders = CONFIG.get("min_traders_per_match", 5)
    filtered_matches = {k: v for k, v in matches.items() if v.total_traders >= min_traders}
    
    # Sort matches by: 1) trader count (most first), 2) consensus strength, 3) volume
    sorted_matches = sorted(
        filtered_matches.values(),
        key=lambda m: (m.total_traders, m.consensus_strength(), m.total_volume),
        reverse=True
    )
    
    # Generate match cards HTML
    match_cards_html = ""
    
    for match in sorted_matches:
        if match.total_traders == 0:
            continue
        
        # Sort outcomes by trader count
        sorted_outcomes = sorted(
            match.outcomes.values(),
            key=lambda o: o.trader_count,
            reverse=True
        )
        
        # Generate outcome bars
        outcomes_html = ""
        for outcome in sorted_outcomes:
            pct = (outcome.trader_count / match.total_traders * 100) if match.total_traders > 0 else 0
            volume_pct = (outcome.total_value / match.total_volume * 100) if match.total_volume > 0 else 0
            
            # Color based on position (green for favorite, others get different shades)
            if outcome == sorted_outcomes[0]:
                bar_color = "#10b981"  # Green for consensus pick
                text_class = "consensus-pick"
            else:
                bar_color = "#6b7280"  # Gray for others
                text_class = ""
            
            outcomes_html += f'''
            <div class="outcome">
                <div class="outcome-header">
                    <span class="outcome-name {text_class}">{outcome.outcome_name}</span>
                    <span class="outcome-stats">{outcome.trader_count} traders · ${outcome.total_value:,.0f}</span>
                </div>
                <div class="bar-container">
                    <div class="bar" style="width: {pct}%; background: {bar_color};">
                        <span class="bar-label">{pct:.1f}%</span>
                    </div>
                </div>
                <div class="volume-bar-container">
                    <div class="volume-bar" style="width: {volume_pct}%;">
                        <span class="volume-label">{volume_pct:.1f}% volume</span>
                    </div>
                </div>
            </div>
            '''
        
        # Format end date
        end_date_display = "TBD"
        if match.end_date:
            try:
                dt = datetime.fromisoformat(match.end_date.replace("Z", "+00:00"))
                end_date_display = dt.strftime("%b %d, %Y %H:%M UTC")
            except:
                end_date_display = match.end_date
        
        # Consensus badge
        consensus = match.consensus_strength()
        if consensus >= 80:
            consensus_class = "consensus-high"
            consensus_label = "Strong Consensus"
        elif consensus >= 60:
            consensus_class = "consensus-medium"
            consensus_label = "Moderate Consensus"
        else:
            consensus_class = "consensus-low"
            consensus_label = "Split Opinion"
        
        match_cards_html += f'''
        <div class="match-card">
            <div class="match-header">
                <div class="match-info">
                    <h3 class="match-title">{match.title}</h3>
                    <div class="match-meta">
                        <span class="end-date">🗓️ {end_date_display}</span>
                        <span class="trader-count">👥 {match.total_traders} top traders</span>
                        <span class="total-volume">💰 ${match.total_volume:,.0f} total</span>
                    </div>
                </div>
                <div class="match-actions">
                    <span class="consensus-badge {consensus_class}">{consensus_label} ({consensus:.0f}%)</span>
                    <a href="{match.polymarket_url}" target="_blank" class="view-market-btn">View on Polymarket →</a>
                </div>
            </div>
            <div class="outcomes">
                {outcomes_html}
            </div>
        </div>
        '''
    
    # Stats summary (using filtered matches)
    total_matches = len([m for m in filtered_matches.values() if m.total_traders > 0])
    high_consensus = len([m for m in filtered_matches.values() if m.consensus_strength() >= 80])
    total_volume = sum(m.total_volume for m in filtered_matches.values())
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polymarket Sports Consensus Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --bg-hover: #22222e;
            --text-primary: #f4f4f5;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --accent-green: #10b981;
            --accent-green-dim: rgba(16, 185, 129, 0.15);
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-orange: #f59e0b;
            --border-color: #27272a;
            --gradient-1: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        .noise-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            opacity: 0.03;
            z-index: 1000;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        /* Navigation */
        .nav {{
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
        }}
        
        .nav-link {{
            color: var(--text-secondary);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: all 0.2s ease;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        
        .nav-link:hover {{
            background: var(--bg-card);
            color: var(--text-primary);
        }}
        
        .nav-link.active {{
            background: var(--accent-green-dim);
            color: var(--accent-green);
        }}
        
        /* Header */
        .header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 3rem 0;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .logo {{
            font-family: 'Space Mono', monospace;
            font-size: 0.875rem;
            color: var(--accent-green);
            text-transform: uppercase;
            letter-spacing: 0.2em;
            margin-bottom: 1rem;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }}
        
        .header p {{
            color: var(--text-secondary);
            font-size: 1.1rem;
        }}
        
        .generated-time {{
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 1rem;
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 3rem;
        }}
        
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.2s ease;
        }}
        
        .stat-card:hover {{
            border-color: var(--accent-green);
            transform: translateY(-2px);
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent-green);
            font-family: 'Space Mono', monospace;
        }}
        
        .stat-label {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }}
        
        /* Filter/Sort Controls */
        .controls {{
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }}
        
        .control-btn {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Outfit', sans-serif;
            font-size: 0.875rem;
            transition: all 0.2s ease;
        }}
        
        .control-btn:hover, .control-btn.active {{
            background: var(--accent-green-dim);
            border-color: var(--accent-green);
            color: var(--accent-green);
        }}
        
        /* Match Cards */
        .matches-container {{
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }}
        
        .match-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.2s ease;
        }}
        
        .match-card:hover {{
            border-color: var(--accent-green);
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.1);
        }}
        
        .match-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1.5rem;
            gap: 1rem;
            flex-wrap: wrap;
        }}
        
        .match-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        .match-meta {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            font-size: 0.875rem;
            color: var(--text-muted);
        }}
        
        .match-actions {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 0.5rem;
        }}
        
        .consensus-badge {{
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            padding: 0.375rem 0.75rem;
            border-radius: 6px;
            font-weight: 600;
        }}
        
        .consensus-high {{
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        
        .consensus-medium {{
            background: rgba(245, 158, 11, 0.2);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}
        
        .consensus-low {{
            background: rgba(107, 114, 128, 0.2);
            color: #9ca3af;
            border: 1px solid rgba(107, 114, 128, 0.3);
        }}
        
        .view-market-btn {{
            font-size: 0.875rem;
            color: var(--accent-blue);
            text-decoration: none;
            transition: color 0.2s ease;
        }}
        
        .view-market-btn:hover {{
            color: var(--accent-green);
        }}
        
        /* Outcomes */
        .outcomes {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}
        
        .outcome {{
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 1rem;
        }}
        
        .outcome-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }}
        
        .outcome-name {{
            font-weight: 500;
            font-size: 1rem;
        }}
        
        .outcome-name.consensus-pick {{
            color: var(--accent-green);
        }}
        
        .outcome-stats {{
            font-size: 0.75rem;
            color: var(--text-muted);
            font-family: 'Space Mono', monospace;
        }}
        
        .bar-container {{
            height: 28px;
            background: var(--bg-primary);
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }}
        
        .bar {{
            height: 100%;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 0.75rem;
            min-width: 50px;
            transition: width 0.5s ease;
        }}
        
        .bar-label {{
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }}
        
        .volume-bar-container {{
            height: 6px;
            background: var(--bg-primary);
            border-radius: 3px;
            overflow: hidden;
        }}
        
        .volume-bar {{
            height: 100%;
            background: var(--accent-purple);
            border-radius: 3px;
            opacity: 0.6;
            position: relative;
        }}
        
        .volume-label {{
            position: absolute;
            right: -60px;
            top: -6px;
            font-size: 0.625rem;
            color: var(--text-muted);
            font-family: 'Space Mono', monospace;
            white-space: nowrap;
        }}
        
        /* Empty State */
        .empty-state {{
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-secondary);
        }}
        
        .empty-state h3 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 3rem 0;
            margin-top: 3rem;
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.875rem;
        }}
        
        .footer a {{
            color: var(--accent-green);
            text-decoration: none;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}
            
            .header h1 {{
                font-size: 1.75rem;
            }}
            
            .match-header {{
                flex-direction: column;
            }}
            
            .match-actions {{
                align-items: flex-start;
            }}
        }}
        
        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .match-card {{
            animation: fadeIn 0.4s ease forwards;
        }}
        
        .match-card:nth-child(1) {{ animation-delay: 0.05s; }}
        .match-card:nth-child(2) {{ animation-delay: 0.1s; }}
        .match-card:nth-child(3) {{ animation-delay: 0.15s; }}
        .match-card:nth-child(4) {{ animation-delay: 0.2s; }}
        .match-card:nth-child(5) {{ animation-delay: 0.25s; }}
    </style>
</head>
<body>
    <div class="noise-overlay"></div>
    
    <div class="container">
        <nav class="nav">
            <a href="index.html" class="nav-link active">📊 Live Predictions</a>
            <a href="analytics.html" class="nav-link">📈 Analytics</a>
        </nav>
        
        <header class="header">
            <div class="logo">⚡ Polymarket Intelligence</div>
            <h1>Sports Consensus Dashboard</h1>
            <p>Where the top 100 profitable traders are placing their bets</p>
            <div class="generated-time">Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC</div>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_matches}</div>
                <div class="stat-label">Active Matches</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{high_consensus}</div>
                <div class="stat-label">High Consensus (80%+)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${total_volume:,.0f}</div>
                <div class="stat-label">Total Volume Tracked</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(traders)}</div>
                <div class="stat-label">Top Traders Analyzed</div>
            </div>
        </div>
        
        <div class="controls">
            <button class="control-btn active" onclick="sortBy('consensus')">Sort by Consensus</button>
            <button class="control-btn" onclick="sortBy('volume')">Sort by Volume</button>
            <button class="control-btn" onclick="sortBy('traders')">Sort by # Traders</button>
            <button class="control-btn" onclick="filterHighConsensus()">High Consensus Only</button>
        </div>
        
        <div class="matches-container" id="matches">
            {match_cards_html if match_cards_html else '<div class="empty-state"><h3>No matches found</h3><p>No sports positions meeting the criteria were found in the next 7 days.</p></div>'}
        </div>
        
        <footer class="footer">
            <p>Data sourced from <a href="https://polymarket.com" target="_blank">Polymarket</a> • Top 100 traders by 30-day profit • Positions ≥$100</p>
            <p style="margin-top: 0.5rem;">This is not financial advice. DYOR.</p>
        </footer>
    </div>
    
    <script>
        // Simple client-side sorting (data is already sorted by consensus)
        function sortBy(criteria) {{
            // Update button states
            document.querySelectorAll('.control-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // In a full implementation, you'd re-sort the cards here
            console.log('Sorting by:', criteria);
        }}
        
        function filterHighConsensus() {{
            event.target.classList.toggle('active');
            const cards = document.querySelectorAll('.match-card');
            const isActive = event.target.classList.contains('active');
            
            cards.forEach(card => {{
                const badge = card.querySelector('.consensus-badge');
                if (isActive && !badge.classList.contains('consensus-high')) {{
                    card.style.display = 'none';
                }} else {{
                    card.style.display = 'block';
                }}
            }});
        }}
    </script>
</body>
</html>'''
    
    return html


def generate_analytics_html(history: PredictionHistoryManager) -> str:
    """Generate analytics dashboard showing prediction accuracy."""
    
    stats = history.get_stats()
    resolved = [p for p in history.predictions if p.resolved]
    pending = [p for p in history.predictions if not p.resolved]
    
    # Sort resolved by date (most recent first)
    resolved_sorted = sorted(resolved, key=lambda p: p.resolved_date or p.recorded_date, reverse=True)
    
    # Generate resolved predictions table
    resolved_html = ""
    for pred in resolved_sorted[:50]:  # Show last 50
        trader_icon = "✅" if pred.trader_consensus_correct else "❌"
        volume_icon = "✅" if pred.volume_consensus_correct else "❌"
        
        # Format date
        try:
            resolved_date = datetime.fromisoformat(pred.resolved_date.replace("Z", "+00:00"))
            date_str = resolved_date.strftime("%b %d, %Y")
        except:
            date_str = "N/A"
        
        resolved_html += f'''
        <tr>
            <td class="game-cell">{pred.title}</td>
            <td class="date-cell">{date_str}</td>
            <td class="outcome-cell winner">{pred.winning_outcome}</td>
            <td class="prediction-cell">
                {trader_icon} {pred.trader_consensus_outcome}<br>
                <span class="prediction-meta">{pred.trader_consensus_count} traders ({pred.trader_consensus_pct:.1f}%)</span>
            </td>
            <td class="prediction-cell">
                {volume_icon} {pred.volume_consensus_outcome}<br>
                <span class="prediction-meta">${pred.volume_consensus_amount:,.0f} ({pred.volume_consensus_pct:.1f}%)</span>
            </td>
        </tr>
        '''
    
    # Calculate trend data (last 30 days)
    recent_cutoff = datetime.utcnow() - timedelta(days=30)
    recent_resolved = [
        p for p in resolved
        if datetime.fromisoformat((p.resolved_date or p.recorded_date).replace("Z", "+00:00")).replace(tzinfo=None) > recent_cutoff
    ]
    
    recent_trader_accuracy = (
        sum(1 for p in recent_resolved if p.trader_consensus_correct) / len(recent_resolved) * 100
        if recent_resolved else 0
    )
    recent_volume_accuracy = (
        sum(1 for p in recent_resolved if p.volume_consensus_correct) / len(recent_resolved) * 100
        if recent_resolved else 0
    )
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analytics - Polymarket Sports Consensus</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --bg-hover: #22222e;
            --text-primary: #f4f4f5;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --accent-green: #10b981;
            --accent-green-dim: rgba(16, 185, 129, 0.15);
            --accent-red: #ef4444;
            --accent-red-dim: rgba(239, 68, 68, 0.15);
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --border-color: #27272a;
            --gradient-1: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        .noise-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            opacity: 0.03;
            z-index: 1000;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        /* Navigation */
        .nav {{
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
        }}
        
        .nav-link {{
            color: var(--text-secondary);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: all 0.2s ease;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        
        .nav-link:hover {{
            background: var(--bg-card);
            color: var(--text-primary);
        }}
        
        .nav-link.active {{
            background: var(--accent-green-dim);
            color: var(--accent-green);
        }}
        
        /* Header */
        .header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem 0;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .logo {{
            font-family: 'Space Mono', monospace;
            font-size: 0.875rem;
            color: var(--accent-green);
            text-transform: uppercase;
            letter-spacing: 0.2em;
            margin-bottom: 1rem;
        }}
        
        .header h1 {{
            font-size: 2rem;
            font-weight: 700;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }}
        
        .header p {{
            color: var(--text-secondary);
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 3rem;
        }}
        
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.2s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-2px);
        }}
        
        .stat-card.winner {{
            border-color: var(--accent-green);
            background: var(--accent-green-dim);
        }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: 700;
            font-family: 'Space Mono', monospace;
            margin-bottom: 0.5rem;
        }}
        
        .stat-value.green {{
            color: var(--accent-green);
        }}
        
        .stat-value.blue {{
            color: var(--accent-blue);
        }}
        
        .stat-value.purple {{
            color: var(--accent-purple);
        }}
        
        .stat-label {{
            font-size: 0.875rem;
            color: var(--text-secondary);
        }}
        
        .stat-sublabel {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        
        /* Comparison Section */
        .comparison {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 3rem;
        }}
        
        .comparison-header {{
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        .comparison-header h2 {{
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .comparison-bars {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            max-width: 800px;
            margin: 0 auto;
        }}
        
        .comparison-item {{
            text-align: center;
        }}
        
        .comparison-label {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }}
        
        .comparison-value {{
            font-size: 3rem;
            font-weight: 700;
            font-family: 'Space Mono', monospace;
            color: var(--accent-green);
            margin-bottom: 0.5rem;
        }}
        
        .comparison-meta {{
            font-size: 0.875rem;
            color: var(--text-muted);
        }}
        
        /* Table */
        .table-container {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 3rem;
        }}
        
        .table-header {{
            padding: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .table-header h2 {{
            font-size: 1.25rem;
            font-weight: 600;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            padding: 1rem;
            text-align: left;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-secondary);
        }}
        
        td {{
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        tr:hover {{
            background: var(--bg-hover);
        }}
        
        .game-cell {{
            font-weight: 500;
            max-width: 300px;
        }}
        
        .date-cell {{
            color: var(--text-muted);
            font-size: 0.875rem;
            font-family: 'Space Mono', monospace;
        }}
        
        .outcome-cell {{
            font-weight: 600;
        }}
        
        .outcome-cell.winner {{
            color: var(--accent-green);
        }}
        
        .prediction-cell {{
            font-size: 0.875rem;
        }}
        
        .prediction-meta {{
            font-size: 0.75rem;
            color: var(--text-muted);
        }}
        
        /* Empty State */
        .empty-state {{
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-secondary);
        }}
        
        .empty-state h3 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 2rem 0;
            margin-top: 3rem;
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.875rem;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .comparison-bars {{
                grid-template-columns: 1fr;
            }}
            
            table {{
                font-size: 0.75rem;
            }}
            
            th, td {{
                padding: 0.75rem 0.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="noise-overlay"></div>
    
    <div class="container">
        <nav class="nav">
            <a href="index.html" class="nav-link">📊 Live Predictions</a>
            <a href="analytics.html" class="nav-link active">📈 Analytics</a>
        </nav>
        
        <header class="header">
            <div class="logo">⚡ Polymarket Intelligence</div>
            <h1>📈 Prediction Accuracy Analytics</h1>
            <p>Tracking: Trader Count Consensus vs. Volume Consensus</p>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value blue">{stats['total_predictions']}</div>
                <div class="stat-label">Total Predictions</div>
                <div class="stat-sublabel">Last 90 days</div>
            </div>
            <div class="stat-card">
                <div class="stat-value green">{stats['resolved']}</div>
                <div class="stat-label">Resolved</div>
            </div>
            <div class="stat-card">
                <div class="stat-value purple">{stats['pending']}</div>
                <div class="stat-label">Pending</div>
            </div>
            <div class="stat-card">
                <div class="stat-value blue">{stats['both_correct']}</div>
                <div class="stat-label">Both Correct</div>
            </div>
        </div>
        
        <div class="comparison">
            <div class="comparison-header">
                <h2>🏆 Accuracy Showdown</h2>
                <p style="color: var(--text-muted); margin-top: 0.5rem;">
                    Which consensus method predicts winners better?
                </p>
            </div>
            
            <div class="comparison-bars">
                <div class="comparison-item">
                    <div class="comparison-label">👥 Trader Count</div>
                    <div class="comparison-value">{stats['trader_accuracy']:.1f}%</div>
                    <div class="comparison-meta">{stats['trader_wins']} / {stats['resolved']} correct</div>
                </div>
                
                <div class="comparison-item">
                    <div class="comparison-label">💰 Volume</div>
                    <div class="comparison-value">{stats['volume_accuracy']:.1f}%</div>
                    <div class="comparison-meta">{stats['volume_wins']} / {stats['resolved']} correct</div>
                </div>
            </div>
        </div>
        
        <div class="table-container">
            <div class="table-header">
                <h2>Recent Results (Last 50)</h2>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Game</th>
                        <th>Date</th>
                        <th>Winner</th>
                        <th>👥 Trader Consensus</th>
                        <th>💰 Volume Consensus</th>
                    </tr>
                </thead>
                <tbody>
                    {resolved_html if resolved_html else '<tr><td colspan="5" class="empty-state"><h3>No resolved predictions yet</h3><p>Check back after games resolve!</p></td></tr>'}
                </tbody>
            </table>
        </div>
        
        <footer class="footer">
            <p>Tracking predictions from top 100 Polymarket traders • Updated every 6 hours</p>
        </footer>
    </div>
</body>
</html>'''
    
    return html


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 60)
    print("🎯 Polymarket Sports Consensus Bot v2 (with Analytics)")
    print("=" * 60)
    print(f"   Leaderboard: Top {CONFIG['leaderboard_limit']} by profit ({CONFIG['leaderboard_window']})")
    print(f"   Min position: ${CONFIG['min_position_size']}")
    print(f"   Lookahead: {CONFIG['days_ahead']} days")
    print(f"   History: {CONFIG['history_days']} days")
    print("=" * 60)
    
    # Initialize
    client = PolymarketClient()
    history = PredictionHistoryManager(CONFIG["history_file"])
    
    try:
        # Step 1: Check for resolved predictions
        update_resolved_predictions(history, client)
        
        # Step 2: Get sports metadata
        client.get_sports_metadata()
        
        # Step 3: Get top traders
        traders = client.get_top_traders(
            window=CONFIG["leaderboard_window"],
            limit=CONFIG["leaderboard_limit"]
        )
        
        if not traders:
            print("❌ No traders found. Check API connection.")
            return
        
        # Step 4: Get sports events/markets
        sports_condition_ids = client.get_sports_events(days_ahead=CONFIG["days_ahead"])
        
        # Step 5: Aggregate positions
        matches = aggregate_positions(
            traders=traders,
            client=client,
            sports_condition_ids=sports_condition_ids,
            min_position_size=CONFIG["min_position_size"],
            days_ahead=CONFIG["days_ahead"],
        )
        
        # Step 6: Add new predictions to history
        print("\n💾 Updating prediction history...")
        for match in matches.values():
            if match.total_traders >= CONFIG.get('min_traders_per_match', 5):
                history.add_prediction(match)
        
        # Step 7: Cleanup old predictions
        history.cleanup_old_predictions(days=CONFIG["history_days"])
        
        # Step 8: Save history
        history.save()
        
        # Step 9: Generate main dashboard HTML
        print("\n🎨 Generating main dashboard...")
        html = generate_html_dashboard(matches, traders)
        output_path = Path(CONFIG["output_file"])
        output_path.write_text(html, encoding="utf-8")
        print(f"✅ Dashboard saved to: {output_path.absolute()}")
        
        # Step 10: Generate analytics HTML
        print("🎨 Generating analytics dashboard...")
        analytics_html = generate_analytics_html(history)
        analytics_path = Path(CONFIG["analytics_file"])
        analytics_path.write_text(analytics_html, encoding="utf-8")
        print(f"✅ Analytics saved to: {analytics_path.absolute()}")
        
        # Step 11: Display stats
        stats = history.get_stats()
        print("\n" + "=" * 60)
        print("📊 PREDICTION ACCURACY STATS")
        print("=" * 60)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Resolved: {stats['resolved']}")
        print(f"Pending: {stats['pending']}")
        if stats['resolved'] > 0:
            print(f"\n👥 Trader Consensus Accuracy: {stats['trader_accuracy']:.1f}% ({stats['trader_wins']}/{stats['resolved']})")
            print(f"💰 Volume Consensus Accuracy: {stats['volume_accuracy']:.1f}% ({stats['volume_wins']}/{stats['resolved']})")
            print(f"🎯 Both Correct: {stats['both_correct']}")
            print(f"❌ Neither Correct: {stats['neither_correct']}")
        print("=" * 60)
        
        # Open in browser only if running locally (not in CI)
        if not os.environ.get("CI") and not os.environ.get("GITHUB_ACTIONS"):
            webbrowser.open(f"file://{output_path.absolute()}")
            print("🌐 Opening dashboard in browser...")
        
    except httpx.HTTPStatusError as e:
        print(f"\n❌ API Error: {e.response.status_code}")
        print(f"   Response: {e.response.text[:200]}")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()
    
    print("\n" + "=" * 60)
    print("✨ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
