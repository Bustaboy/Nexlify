# /home/netrunner/neuralink_project/utils/data_stream.py
"""
Data Stream Manager - The neural system's sensory network
Handles real-time data ingestion from the sprawl
Simulates connection to market feeds, urban sensors, and crowd monitors
"""

import numpy as np
import time
import threading
import queue
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import json

class DataStreamManager:
    """
    Manages multiple real-time data streams
    In production, would connect to actual data sources
    For prototype, generates realistic synthetic data
    """
    
    def __init__(self, stream_config: Dict):
        self.config = stream_config
        self.running = False
        
        # Data queues for each stream
        self.market_queue = queue.Queue(maxsize=1000)
        self.crowd_queue = queue.Queue(maxsize=1000)
        self.urban_queue = queue.Queue(maxsize=1000)
        
        # Stream threads
        self.threads = {}
        
        # Data generators
        self.market_generator = MarketDataGenerator()
        self.crowd_generator = CrowdDataGenerator()
        self.urban_generator = UrbanDataGenerator()
        
        # Statistics
        self.stats = {
            'market_messages': 0,
            'crowd_messages': 0,
            'urban_messages': 0,
            'dropped_messages': 0
        }
        
    def start(self):
        """Start all data streams"""
        self.running = True
        
        # Start market stream
        self.threads['market'] = threading.Thread(
            target=self._market_stream_worker,
            daemon=True
        )
        self.threads['market'].start()
        
        # Start crowd stream
        self.threads['crowd'] = threading.Thread(
            target=self._crowd_stream_worker,
            daemon=True
        )
        self.threads['crowd'].start()
        
        # Start urban stream
        self.threads['urban'] = threading.Thread(
            target=self._urban_stream_worker,
            daemon=True
        )
        self.threads['urban'].start()
        
    def _market_stream_worker(self):
        """Worker thread for market data stream"""
        while self.running:
            try:
                # Generate market data
                data = self.market_generator.generate_batch(10)
                
                for item in data:
                    try:
                        self.market_queue.put(item, timeout=0.1)
                        self.stats['market_messages'] += 1
                    except queue.Full:
                        self.stats['dropped_messages'] += 1
                
                # Simulate real-time delay
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[ERROR] Market stream error: {e}")
    
    def _crowd_stream_worker(self):
        """Worker thread for crowd data stream"""
        while self.running:
            try:
                # Generate crowd data
                data = self.crowd_generator.generate_batch(20)
                
                for item in data:
                    try:
                        self.crowd_queue.put(item, timeout=0.1)
                        self.stats['crowd_messages'] += 1
                    except queue.Full:
                        self.stats['dropped_messages'] += 1
                
                # Simulate real-time delay
                time.sleep(0.2)
                
            except Exception as e:
                print(f"[ERROR] Crowd stream error: {e}")
    
    def _urban_stream_worker(self):
        """Worker thread for urban data stream"""
        while self.running:
            try:
                # Generate urban data
                data = self.urban_generator.generate_batch(15)
                
                for item in data:
                    try:
                        self.urban_queue.put(item, timeout=0.1)
                        self.stats['urban_messages'] += 1
                    except queue.Full:
                        self.stats['dropped_messages'] += 1
                
                # Simulate real-time delay
                time.sleep(0.15)
                
            except Exception as e:
                print(f"[ERROR] Urban stream error: {e}")
    
    def get_market_data(self, max_items: int = 100) -> List[Dict]:
        """Get available market data"""
        data = []
        while len(data) < max_items and not self.market_queue.empty():
            try:
                item = self.market_queue.get_nowait()
                data.append(item)
            except queue.Empty:
                break
        return data
    
    def get_crowd_data(self, max_items: int = 100) -> List[Dict]:
        """Get available crowd data"""
        data = []
        while len(data) < max_items and not self.crowd_queue.empty():
            try:
                item = self.crowd_queue.get_nowait()
                data.append(item)
            except queue.Empty:
                break
        return data
    
    def get_urban_data(self, max_items: int = 100) -> List[Dict]:
        """Get available urban data"""
        data = []
        while len(data) < max_items and not self.urban_queue.empty():
            try:
                item = self.urban_queue.get_nowait()
                data.append(item)
            except queue.Empty:
                break
        return data
    
    def get_stats(self) -> Dict:
        """Get stream statistics"""
        return {
            **self.stats,
            'market_queue_size': self.market_queue.qsize(),
            'crowd_queue_size': self.crowd_queue.qsize(),
            'urban_queue_size': self.urban_queue.qsize()
        }
    
    def close(self):
        """Close all data streams"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads.values():
            thread.join(timeout=1.0)

class MarketDataGenerator:
    """Generates realistic market data for testing"""
    
    def __init__(self):
        self.base_price = 1000.0
        self.volatility = 0.02
        self.trend = 0.0001
        self.manipulation_active = False
        self.manipulation_start = 0
        
    def generate_batch(self, size: int) -> List[Dict]:
        """Generate batch of market data"""
        data = []
        current_time = time.time()
        
        for i in range(size):
            # Simulate price movement
            price_change = np.random.normal(self.trend, self.volatility)
            self.base_price *= (1 + price_change)
            
            # Random volume
            volume = np.random.lognormal(10, 1)
            
            # Calculate volatility
            volatility = self.volatility * (1 + 0.1 * np.sin(current_time / 3600))
            
            # Sentiment (-1 to 1)
            sentiment = np.tanh(np.random.normal(0, 0.5))
            
            # Manipulation detection
            if np.random.random() < 0.01:  # 1% chance to start manipulation
                self.manipulation_active = True
                self.manipulation_start = current_time
            
            if self.manipulation_active and current_time - self.manipulation_start > 300:
                self.manipulation_active = False
            
            manipulation_score = 0.8 if self.manipulation_active else np.random.random() * 0.2
            
            data.append({
                'timestamp': current_time + i * 0.01,
                'price': self.base_price,
                'volume': volume,
                'volatility': volatility,
                'sentiment': sentiment,
                'manipulation_score': manipulation_score
            })
        
        return data

class CrowdDataGenerator:
    """Generates realistic crowd behavior data"""
    
    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        self.base_density = np.random.rand(grid_size, grid_size) * 50
        self.crowd_state = 'dormant'
        self.state_duration = 0
        
    def generate_batch(self, size: int) -> List[Dict]:
        """Generate batch of crowd data"""
        data = []
        current_time = time.time()
        
        # Update crowd state
        self.state_duration += 1
        if self.state_duration > np.random.randint(100, 500):
            self._transition_state()
            self.state_duration = 0
        
        for i in range(size):
            # Random location
            location_id = np.random.randint(0, self.grid_size * self.grid_size)
            x = location_id // self.grid_size
            y = location_id % self.grid_size
            
            # Base density with noise
            base = self.base_density[x, y]
            density = base * (1 + np.random.normal(0, 0.1))
            
            # Movement variance based on state
            movement_multiplier = {
                'dormant': 0.1,
                'agitated': 0.5,
                'volatile': 0.8,
                'erupting': 1.0
            }.get(self.crowd_state, 0.5)
            
            movement_variance = np.random.exponential(1) * movement_multiplier
            
            # Other metrics
            noise_level = 60 + density * 0.5 + movement_variance * 10
            social_sentiment = np.tanh(np.random.normal(0, 0.5))
            temperature = 20 + 10 * np.sin(current_time / 86400) + np.random.normal(0, 2)
            police_presence = min(1, max(0, movement_variance * 0.5 + np.random.normal(0, 0.1)))
            economic_stress = 0.3 + 0.2 * np.sin(current_time / (86400 * 30)) + np.random.normal(0, 0.1)
            
            data.append({
                'timestamp': current_time + i * 0.01,
                'location_id': location_id,
                'density': density,
                'movement_variance': movement_variance,
                'noise_level': noise_level,
                'social_media_sentiment': social_sentiment,
                'temperature': temperature,
                'police_presence': police_presence,
                'economic_stress': np.clip(economic_stress, 0, 1)
            })
        
        return data
    
    def _transition_state(self):
        """Transition crowd state"""
        transitions = {
            'dormant': ['dormant', 'agitated'],
            'agitated': ['dormant', 'agitated', 'volatile'],
            'volatile': ['agitated', 'volatile', 'erupting'],
            'erupting': ['volatile', 'erupting', 'dispersing'],
            'dispersing': ['dormant', 'agitated']
        }
        
        possible_states = transitions.get(self.crowd_state, ['dormant'])
        self.crowd_state = np.random.choice(possible_states)

class UrbanDataGenerator:
    """Generates realistic urban infrastructure data"""
    
    def __init__(self, grid_size: int = 128):
        self.grid_size = grid_size
        self.time_of_day = 0
        self.day_of_week = 0
        
        # Initialize zone map
        self.zone_map = self._create_zone_map()
        
    def _create_zone_map(self) -> np.ndarray:
        """Create urban zone map"""
        zone_map = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Simple zone layout
        center = self.grid_size // 2
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                
                if dist < self.grid_size * 0.1:
                    zone_map[i, j] = 0  # Corpo plaza
                elif dist < self.grid_size * 0.2:
                    zone_map[i, j] = 1  # Market district
                elif dist < self.grid_size * 0.3:
                    zone_map[i, j] = 2  # Industrial
                elif dist < self.grid_size * 0.4:
                    zone_map[i, j] = 3  # Residential
                else:
                    zone_map[i, j] = 4  # Slums
        
        return zone_map
    
    def generate_batch(self, size: int) -> List[Dict]:
        """Generate batch of urban data"""
        data = []
        current_time = time.time()
        
        # Update time of day
        self.time_of_day = (current_time % 86400) / 3600  # Hour of day
        self.day_of_week = int((current_time // 86400) % 7)
        
        for i in range(size):
            # Random zone
            zone_id = np.random.randint(0, self.grid_size * self.grid_size)
            x = zone_id // self.grid_size
            y = zone_id % self.grid_size
            zone_type = self.zone_map[x, y]
            
            # Traffic patterns based on time and zone
            traffic_multiplier = self._get_traffic_pattern(zone_type, self.time_of_day)
            traffic_density = np.random.exponential(100) * traffic_multiplier
            
            # Pedestrian flow
            pedestrian_flow = np.random.exponential(50) * traffic_multiplier * 0.7
            
            # Resource consumption based on zone and time
            power_base = [500, 300, 800, 200, 100][zone_type]
            power_consumption = power_base * (1 + 0.3 * np.sin(self.time_of_day * np.pi / 12))
            
            water_base = [200, 150, 400, 300, 50][zone_type]
            water_usage = water_base * (1 + 0.2 * np.sin(self.time_of_day * np.pi / 12))
            
            # Environmental metrics
            air_quality = 0.8 - 0.3 * (traffic_density / 1000)
            noise_pollution = 50 + traffic_density * 0.05 + pedestrian_flow * 0.02
            
            # Economic activity
            economic_activity = traffic_multiplier * pedestrian_flow * 0.01
            
            # Crime index (higher at night in certain zones)
            crime_multiplier = 1 + 0.5 * np.sin((self.time_of_day - 2) * np.pi / 12)
            crime_base = [0.1, 0.2, 0.3, 0.4, 0.6][zone_type]
            crime_index = crime_base * crime_multiplier
            
            data.append({
                'timestamp': current_time + i * 0.01,
                'zone_id': zone_id,
                'traffic_density': traffic_density,
                'pedestrian_flow': pedestrian_flow,
                'power_consumption': power_consumption,
                'water_usage': water_usage,
                'waste_generation': (power_consumption + water_usage) * 0.001,
                'air_quality': np.clip(air_quality, 0, 1),
                'noise_pollution': noise_pollution,
                'crime_index': np.clip(crime_index, 0, 1),
                'economic_activity': economic_activity
            })
        
        return data
    
    def _get_traffic_pattern(self, zone_type: int, hour: float) -> float:
        """Get traffic multiplier based on zone and time"""
        # Rush hour patterns
        morning_rush = np.exp(-((hour - 8)**2) / 4)
        evening_rush = np.exp(-((hour - 18)**2) / 4)
        night_life = np.exp(-((hour - 22)**2) / 8) if zone_type in [1, 4] else 0
        
        # Zone-specific patterns
        if zone_type == 0:  # Corpo plaza
            return 0.5 + morning_rush + evening_rush
        elif zone_type == 1:  # Market district
            return 0.7 + 0.5 * morning_rush + 0.5 * evening_rush + night_life
        elif zone_type == 2:  # Industrial
            return 0.3 + 0.8 * morning_rush + 0.8 * evening_rush
        elif zone_type == 3:  # Residential
            return 0.2 + 0.6 * morning_rush + 0.6 * evening_rush
        else:  # Slums
            return 0.1 + 0.2 * np.random.random() + night_life
        
class StreamRecorder:
    """Records data streams for replay and analysis"""
    
    def __init__(self, output_dir: str = "data/recordings"):
        self.output_dir = output_dir
        self.recording = False
        self.buffers = {
            'market': [],
            'crowd': [],
            'urban': []
        }
        self.start_time = None
        
    def start_recording(self):
        """Start recording data streams"""
        self.recording = True
        self.start_time = time.time()
        self.buffers = {'market': [], 'crowd': [], 'urban': []}
        
    def record_data(self, stream_type: str, data: List[Dict]):
        """Record data from a stream"""
        if self.recording and stream_type in self.buffers:
            self.buffers[stream_type].extend(data)
    
    def stop_recording(self) -> str:
        """Stop recording and save data"""
        self.recording = False
        
        if not self.start_time:
            return None
        
        # Create filename with timestamp
        timestamp = int(self.start_time)
        filename = f"{self.output_dir}/recording_{timestamp}.json"
        
        # Save recording
        recording_data = {
            'start_time': self.start_time,
            'duration': time.time() - self.start_time,
            'streams': self.buffers
        }
        
        with open(filename, 'w') as f:
            json.dump(recording_data, f)
        
        return filename

class StreamReplayer:
    """Replays recorded data streams"""
    
    def __init__(self, recording_file: str):
        self.recording_file = recording_file
        self.recording_data = None
        self.playback_position = {'market': 0, 'crowd': 0, 'urban': 0}
        
        self._load_recording()
        
    def _load_recording(self):
        """Load recording from file"""
        with open(self.recording_file, 'r') as f:
            self.recording_data = json.load(f)
    
    def get_next_batch(self, stream_type: str, batch_size: int) -> List[Dict]:
        """Get next batch of data from recording"""
        if not self.recording_data or stream_type not in self.recording_data['streams']:
            return []
        
        stream_data = self.recording_data['streams'][stream_type]
        start_pos = self.playback_position[stream_type]
        end_pos = min(start_pos + batch_size, len(stream_data))
        
        batch = stream_data[start_pos:end_pos]
        
        # Update position
        self.playback_position[stream_type] = end_pos
        
        # Loop if reached end
        if end_pos >= len(stream_data):
            self.playback_position[stream_type] = 0
        
        return batch
    
    def reset(self):
        """Reset playback to beginning"""
        self.playback_position = {'market': 0, 'crowd': 0, 'urban': 0}
