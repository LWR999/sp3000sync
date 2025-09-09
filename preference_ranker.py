#!/usr/bin/env python3
# preference_ranker.py
# A module for ranking music albums based on jazz/electronic/lounge preferences

import sqlite3
import json
import os
import logging
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

# Try to import optional dependencies with helpful error messages
try:
    import requests
except ImportError:
    raise ImportError(
        "The 'requests' package is required but not installed. "
        "Please install it using: pip install requests"
    )

try:
    import yaml
except ImportError:
    raise ImportError(
        "The 'pyyaml' package is required but not installed. "
        "Please install it using: pip install pyyaml"
    )

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    raise ImportError(
        "The 'scikit-learn' package is required but not installed. "
        "Please install it using: pip install scikit-learn"
    )


class PreferenceRanker:
    def __init__(self, config_path, log_path=None, profile_name=None, verbose=False):
        """Initialize the ranker with configuration
        
        Args:
            config_path (str): Path to the configuration file
            log_path (str, optional): Path to the log file. If None, logs to console.
            profile_name (str, optional): Name of the preference profile to use.
                If None, uses the default profile from config.
            verbose (bool, optional): Whether to enable verbose logging.
        """
        # Setup logging first
        log_level = logging.DEBUG if verbose else logging.INFO
        if log_path:
            logging.basicConfig(filename=log_path, level=log_level,
                               format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=log_level,
                               format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('preference_ranker')
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    self.config = json.load(f)
                else:
                    # Try YAML first, then JSON if that fails
                    try:
                        self.config = yaml.safe_load(f)
                    except yaml.YAMLError:
                        # Rewind file and try JSON
                        f.seek(0)
                        self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
        
        # Load preferences
        self.profile_name = profile_name or self.config.get('music_preferences', {}).get('default_profile')
        if not self.profile_name:
            self.logger.warning("No profile specified and no default profile in config. Using standard references.")
            self.profile = None
        else:
            try:
                self.profile = self.config['music_preferences']['profiles'][self.profile_name]
                self.logger.info(f"Loaded music preference profile: {self.profile_name}")
            except KeyError:
                self.logger.error(f"Profile '{self.profile_name}' not found in config. Using standard references.")
                self.profile = None
        
        # Initialize other config variables
        self.db_path = self.config.get('paths', {}).get('database')
        if not self.db_path:
            self.logger.error("Database path not found in configuration")
            raise ValueError("Database path is required in configuration")
            
        llm_config = self.config.get('llm', {})
        self.api_key = llm_config.get('api_key', '')
        self.api_url = llm_config.get('api_url', 'https://api.anthropic.com/v1/messages')
        self.embedding_url = llm_config.get('embedding_url', 'https://api.anthropic.com/v1/embeddings')
        self.model = llm_config.get('model', 'claude-3-sonnet-20240229')
        self.embedding_model = llm_config.get('embedding_model', 'claude-3-sonnet-20240229-embedding')
        
        # Set up embedding cache to avoid repeated API calls
        self.embedding_cache = {}
        self.cache_dir = self.config.get('paths', {}).get('cache')
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.embedding_cache_file = os.path.join(self.cache_dir, 'embedding_cache.json')
            self._load_embedding_cache()
        else:
            self.embedding_cache_file = None
        
        if not self.api_key:
            self.logger.warning("API key not provided. LLM functionality will not work.")
    
    def _load_embedding_cache(self):
        """Load embedding cache from file if it exists"""
        if not self.embedding_cache_file:
            return
        
        try:
            if os.path.exists(self.embedding_cache_file):
                with open(self.embedding_cache_file, 'r') as f:
                    self.embedding_cache = json.load(f)
                self.logger.info(f"Loaded {len(self.embedding_cache)} embeddings from cache")
        except Exception as e:
            self.logger.warning(f"Failed to load embedding cache: {str(e)}")
            self.embedding_cache = {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to file"""
        if not self.embedding_cache_file:
            return
        
        try:
            with open(self.embedding_cache_file, 'w') as f:
                json.dump(self.embedding_cache, f)
            self.logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            self.logger.warning(f"Failed to save embedding cache: {str(e)}")
    
    def connect_db(self):
        """Establish a connection to the SQLite database"""
        try:
            return sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {str(e)}")
            raise
    
    def get_embedding(self, text):
        """Get embedding vector for text using Claude API
        
        Args:
            text (str): The text to get an embedding for
            
        Returns:
            list: The embedding vector, or None if the request failed
        """
        # Check cache first
        cache_key = text.strip().lower()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        if not self.api_key:
            self.logger.error("API key not configured. Please add it to your config file.")
            return None
            
        try:
            response = requests.post(
                self.embedding_url,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key
                },
                json={
                    "model": self.embedding_model,
                    "input": text
                },
                timeout=30
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            embedding = response.json().get("embedding")
            
            # Cache the result
            if embedding:
                self.embedding_cache[cache_key] = embedding
                # Periodically save cache
                if len(self.embedding_cache) % 10 == 0:
                    self._save_embedding_cache()
                    
            return embedding
            
        except requests.RequestException as e:
            self.logger.error(f"API request error: {str(e)}")
            return None
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.logger.error(f"Error parsing API response: {str(e)}")
            return None
    
    def create_reference_embeddings(self):
        """Create reference embeddings from config file preferences"""
        # Fallback reference points if none in config
        default_reference_points = [
            "Jazz fusion with electronic elements and smooth textures",
            "Downtempo electronic music with jazz instrumentation",
            "Nu-jazz with atmospheric electronic production",
            "Lounge music with jazz samples and electronic beats",
            "Acid jazz with electronic synthesizers and dance elements",
            "Chill-out electronic music with jazz influences",
            "Electronic lounge music with improvisational jazz solos"
        ]
        
        # Get reference points from config or use defaults
        if self.profile and 'reference_points' in self.profile:
            reference_points = self.profile['reference_points']
            self.logger.info(f"Using {len(reference_points)} reference points from profile '{self.profile_name}'")
        else:
            reference_points = default_reference_points
            self.logger.info(f"Using {len(reference_points)} default reference points")
        
        # Generate embeddings
        embeddings = []
        for point in reference_points:
            embedding = self.get_embedding(point)
            if embedding:
                embeddings.append(embedding)
            else:
                self.logger.warning(f"Failed to get embedding for: {point}")
        
        if not embeddings:
            self.logger.error("Failed to create any reference embeddings")
        
        return embeddings
    
    def create_artist_specific_embeddings(self):
        """Create artist-specific reference embeddings from config"""
        # Default artist references if none in config
        default_artist_references = [
            "Music similar to Soulpersona's modern soul with vintage production techniques",
            "Albums that sound like Brand New Heavies with their acid jazz, funk basslines and horn sections"
        ]
        
        # Get artist references from config or use defaults
        if self.profile and 'artist_references' in self.profile:
            artist_references = self.profile['artist_references']
            self.logger.info(f"Using {len(artist_references)} artist references from profile '{self.profile_name}'")
        else:
            artist_references = default_artist_references
            self.logger.info(f"Using {len(artist_references)} default artist references")
        
        # Generate embeddings
        embeddings = []
        for reference in artist_references:
            embedding = self.get_embedding(reference)
            if embedding:
                embeddings.append(embedding)
            else:
                self.logger.warning(f"Failed to get embedding for: {reference}")
        
        if not embeddings:
            self.logger.error("Failed to create any artist-specific reference embeddings")
        
        return embeddings
    
    def create_weighted_reference_embeddings(self):
        """Create weighted reference embeddings from config"""
        # Default weighted references if none in config
        default_weighted_refs = [
            {"reference": "Jazz fusion with electronic elements and smooth textures", "weight": 2.0},
            {"reference": "Acid jazz with electronic synthesizers and dance elements", "weight": 2.0}
        ]
        
        # Get weighted references from config or use defaults
        if self.profile and 'weighted_references' in self.profile:
            weighted_refs = self.profile['weighted_references']
            self.logger.info(f"Using {len(weighted_refs)} weighted references from profile '{self.profile_name}'")
        else:
            weighted_refs = default_weighted_refs
            self.logger.info(f"Using {len(weighted_refs)} default weighted references")
        
        # Generate embeddings
        weighted_embeddings = []
        for item in weighted_refs:
            # Handle both list format and dict format
            if isinstance(item, dict):
                reference = item.get('reference')
                weight = float(item.get('weight', 1.0))
            else:
                reference = item[0] if len(item) > 0 else None
                weight = float(item[1]) if len(item) > 1 else 1.0
                
            if not reference:
                continue
                
            embedding = self.get_embedding(reference)
            if embedding:
                weighted_embeddings.append((embedding, weight))
            else:
                self.logger.warning(f"Failed to get embedding for: {reference}")
        
        if not weighted_embeddings:
            self.logger.error("Failed to create any weighted reference embeddings")
        
        return weighted_embeddings
    
    def fetch_albums(self, limit=None, offset=0):
        """Fetch albums with their metadata from the database
        
        Args:
            limit (int, optional): Maximum number of albums to fetch
            offset (int, optional): Number of albums to skip
            
        Returns:
            list: List of album tuples (id, artist, title, track_titles)
        """
        conn = self.connect_db()
        cursor = conn.cursor()
        
        query = """
        SELECT 
            albums.id AS album_id, 
            artists.name AS artist_name,
            albums.title AS album_title,
            GROUP_CONCAT(tracks.title, ' | ') AS track_titles
        FROM 
            albums
        JOIN 
            artists ON albums.artist_id = artists.id
        JOIN 
            tracks ON tracks.album_id = albums.id
        GROUP BY 
            albums.id
        """
        
        # Add limit and offset if specified
        if limit is not None:
            query += f" LIMIT {limit}"
        if offset > 0:
            query += f" OFFSET {offset}"
        
        try:
            cursor.execute(query)
            albums = cursor.fetchall()
            self.logger.info(f"Fetched {len(albums)} albums from database")
            return albums
        except sqlite3.Error as e:
            self.logger.error(f"Database query error: {str(e)}")
            return []
        finally:
            conn.close()
    
    def calculate_similarity(self, album_embedding, reference_embeddings):
        """Calculate cosine similarity between album and reference embeddings
        
        Args:
            album_embedding (list): The album embedding vector
            reference_embeddings (list): List of reference embedding vectors
            
        Returns:
            tuple: (max_similarity, best_match_index)
        """
        # Calculate similarity to each reference point
        similarities = []
        for ref_emb in reference_embeddings:
            similarity = cosine_similarity([album_embedding], [ref_emb])[0][0]
            similarities.append(similarity)
        
        # Take the maximum similarity (closest match to any reference point)
        max_similarity = max(similarities)
        best_match_index = similarities.index(max_similarity)
        
        return max_similarity, best_match_index
    
    def calculate_similarity_weighted(self, album_embedding, weighted_embeddings):
        """Calculate weighted cosine similarity between album and reference embeddings
        
        Args:
            album_embedding (list): The album embedding vector
            weighted_embeddings (list): List of (embedding, weight) tuples
            
        Returns:
            tuple: (max_weighted_similarity, best_match_index)
        """
        # Calculate similarity to each reference point and apply weights
        weighted_similarities = []
        for ref_emb, weight in weighted_embeddings:
            similarity = cosine_similarity([album_embedding], [ref_emb])[0][0]
            weighted_similarity = similarity * weight
            weighted_similarities.append(weighted_similarity)
        
        # Take the maximum weighted similarity
        max_weighted_similarity = max(weighted_similarities)
        best_match_index = weighted_similarities.index(max_weighted_similarity)
        
        return max_weighted_similarity, best_match_index
    
    def rank_albums(self, method='combined', batch_size=None, max_albums=None):
        """
        Rank all albums based on similarity to preferences
        
        Args:
            method (str): The ranking method to use:
                          'standard' - use regular reference points
                          'artist' - use artist-specific references
                          'weighted' - use weighted references
                          'combined' - use all methods and combine scores (default)
            batch_size (int, optional): Size of batches to process (for memory efficiency)
            max_albums (int, optional): Maximum number of albums to process
            
        Returns:
            list: List of tuples (album_id, artist, title, score) sorted by score
        """
        start_time = time.time()
        album_scores = []
        
        # Process in batches if batch_size is specified
        if batch_size:
            return self.rank_albums_batched(method, batch_size, max_albums)
        
        # Get albums
        albums = self.fetch_albums(max_albums)
        if not albums:
            self.logger.error("No albums found in database")
            return []
        
        self.logger.info(f"Processing {len(albums)} albums using {method} method")
        
        if method in ['standard', 'combined']:
            # Get standard reference embeddings
            ref_embeddings = self.create_reference_embeddings()
            if not ref_embeddings:
                self.logger.error("Failed to create standard reference embeddings")
                if method == 'standard':
                    return []
            else:
                # Process with standard embeddings
                standard_scores = self._process_albums_with_embeddings(albums, ref_embeddings)
                if method == 'standard':
                    elapsed = time.time() - start_time
                    self.logger.info(f"Ranking completed in {elapsed:.2f} seconds")
                    return standard_scores
        
        if method in ['artist', 'combined']:
            # Get artist-specific reference embeddings
            artist_embeddings = self.create_artist_specific_embeddings()
            if not artist_embeddings:
                self.logger.error("Failed to create artist-specific reference embeddings")
                if method == 'artist':
                    return []
            else:
                # Process with artist embeddings
                artist_scores = self._process_albums_with_embeddings(albums, artist_embeddings)
                if method == 'artist':
                    elapsed = time.time() - start_time
                    self.logger.info(f"Ranking completed in {elapsed:.2f} seconds")
                    return artist_scores
        
        if method in ['weighted', 'combined']:
            # Get weighted reference embeddings
            weighted_embeddings = self.create_weighted_reference_embeddings()
            if not weighted_embeddings:
                self.logger.error("Failed to create weighted reference embeddings")
                if method == 'weighted':
                    return []
            else:
                # Process with weighted embeddings
                weighted_scores = self._process_albums_with_weighted_embeddings(albums, weighted_embeddings)
                if method == 'weighted':
                    elapsed = time.time() - start_time
                    self.logger.info(f"Ranking completed in {elapsed:.2f} seconds")
                    return weighted_scores
        
        if method == 'combined':
            # Combine scores from all methods
            album_map = {}
            
            # Helper to add scores to the map
            def add_scores(scores, weight=1.0):
                for album_id, artist, title, score in scores:
                    key = (album_id, artist, title)
                    if key in album_map:
                        album_map[key] += score * weight
                    else:
                        album_map[key] = score * weight
            
            # Add all scores with appropriate weights
            if 'standard_scores' in locals():
                add_scores(standard_scores, 1.0)
            if 'artist_scores' in locals():
                add_scores(artist_scores, 1.5)  # Artist-specific gets higher weight
            if 'weighted_scores' in locals():
                add_scores(weighted_scores, 2.0)  # Weighted gets highest priority
            
            # Convert map back to list format
            combined_scores = [(album_id, artist, title, score) 
                               for (album_id, artist, title), score in album_map.items()]
            
            # Sort by combined score
            combined_scores.sort(key=lambda x: x[3], reverse=True)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Ranking completed in {elapsed:.2f} seconds")
            return combined_scores
        
        # Fallback
        self.logger.error(f"Unknown ranking method: {method}")
        return []
    
    def rank_albums_batched(self, method='combined', batch_size=50, max_albums=None):
        """
        Rank albums in batches to conserve memory
        
        Args:
            method (str): The ranking method to use
            batch_size (int): Number of albums to process in each batch
            max_albums (int, optional): Maximum number of albums to process
            
        Returns:
            list: List of tuples (album_id, artist, title, score) sorted by score
        """
        start_time = time.time()
        
        # Connect to database to get total count
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM albums")
            total_albums = cursor.fetchone()[0]
            
            if max_albums is not None and max_albums < total_albums:
                total_albums = max_albums
                
            self.logger.info(f"Processing {total_albums} albums in batches of {batch_size}")
        except sqlite3.Error as e:
            self.logger.error(f"Error getting album count: {str(e)}")
            conn.close()
            return []
        finally:
            conn.close()
        
        # Initialize embeddings for each method
        embeddings = {}
        if method in ['standard', 'combined']:
            embeddings['standard'] = self.create_reference_embeddings()
        if method in ['artist', 'combined']:
            embeddings['artist'] = self.create_artist_specific_embeddings()
        if method in ['weighted', 'combined']:
            embeddings['weighted'] = self.create_weighted_reference_embeddings()
        
        # Check if we have any embeddings
        if not any(embeddings.values()):
            self.logger.error("Failed to create any reference embeddings")
            return []
        
        # Process in batches
        all_scores = {
            'standard': [],
            'artist': [],
            'weighted': []
        }
        
        offset = 0
        while offset < total_albums:
            # Adjust batch size for last batch
            current_batch_size = min(batch_size, total_albums - offset)
            if current_batch_size <= 0:
                break
                
            # Fetch batch
            albums = self.fetch_albums(current_batch_size, offset)
            if not albums:
                break
                
            self.logger.info(f"Processing batch {offset//batch_size + 1}, albums {offset+1}-{offset+len(albums)}")
            
            # Process with each embedding type
            if 'standard' in embeddings and embeddings['standard']:
                batch_scores = self._process_albums_with_embeddings(albums, embeddings['standard'])
                all_scores['standard'].extend(batch_scores)
                
            if 'artist' in embeddings and embeddings['artist']:
                batch_scores = self._process_albums_with_embeddings(albums, embeddings['artist'])
                all_scores['artist'].extend(batch_scores)
                
            if 'weighted' in embeddings and embeddings['weighted']:
                batch_scores = self._process_albums_with_weighted_embeddings(albums, embeddings['weighted'])
                all_scores['weighted'].extend(batch_scores)
            
            offset += current_batch_size
        
        # Return based on method
        if method == 'standard':
            scores = all_scores['standard']
        elif method == 'artist':
            scores = all_scores['artist']
        elif method == 'weighted':
            scores = all_scores['weighted']
        elif method == 'combined':
            # Combine scores from all methods
            album_map = {}
            
            # Helper to add scores to the map
            def add_scores(scores, weight=1.0):
                for album_id, artist, title, score in scores:
                    key = (album_id, artist, title)
                    if key in album_map:
                        album_map[key] += score * weight
                    else:
                        album_map[key] = score * weight
            
            # Add all scores with appropriate weights
            add_scores(all_scores['standard'], 1.0)
            add_scores(all_scores['artist'], 1.5)  # Artist-specific gets higher weight
            add_scores(all_scores['weighted'], 2.0)  # Weighted gets highest priority
            
            # Convert map back to list format
            scores = [(album_id, artist, title, score) 
                      for (album_id, artist, title), score in album_map.items()]
        else:
            self.logger.error(f"Unknown ranking method: {method}")
            return []
        
        # Sort by score
        scores.sort(key=lambda x: x[3], reverse=True)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Ranking completed in {elapsed:.2f} seconds")
        return scores
    
    def _process_albums_with_embeddings(self, albums, ref_embeddings):
        """Process all albums with standard embeddings
        
        Args:
            albums (list): List of album tuples
            ref_embeddings (list): List of reference embeddings
            
        Returns:
            list: List of (album_id, artist, title, score) tuples
        """
        album_scores = []
        
        for i, (album_id, artist, title, track_titles) in enumerate(albums):
            # Log progress periodically
            if i % 20 == 0:
                self.logger.debug(f"Processing album {i+1}/{len(albums)}")
                
            # Create album description
            description = f"Artist: {artist}. Album: {title}. Tracks: {track_titles}"
            
            # Get embedding
            album_embedding = self.get_embedding(description)
            if not album_embedding:
                self.logger.warning(f"Skipping album: {artist} - {title} (failed to get embedding)")
                continue
                
            # Calculate similarity
            similarity, _ = self.calculate_similarity(album_embedding, ref_embeddings)
            album_scores.append((album_id, artist, title, similarity))
            self.logger.debug(f"Album: {artist} - {title} - Score: {similarity:.4f}")
        
        # Sort by similarity
        album_scores.sort(key=lambda x: x[3], reverse=True)
        return album_scores
    
    def _process_albums_with_weighted_embeddings(self, albums, weighted_embeddings):
        """Process all albums with weighted embeddings
        
        Args:
            albums (list): List of album tuples
            weighted_embeddings (list): List of (embedding, weight) tuples
            
        Returns:
            list: List of (album_id, artist, title, score) tuples
        """
        album_scores = []
        
        for i, (album_id, artist, title, track_titles) in enumerate(albums):
            # Log progress periodically
            if i % 20 == 0:
                self.logger.debug(f"Processing album {i+1}/{len(albums)}")
                
            # Create album description
            description = f"Artist: {artist}. Album: {title}. Tracks: {track_titles}"
            
            # Get embedding
            album_embedding = self.get_embedding(description)
            if not album_embedding:
                self.logger.warning(f"Skipping album: {artist} - {title} (failed to get embedding)")
                continue
                
            # Calculate weighted similarity
            similarity, _ = self.calculate_similarity_weighted(album_embedding, weighted_embeddings)
            album_scores.append((album_id, artist, title, similarity))
            self.logger.debug(f"Album: {artist} - {title} - Weighted Score: {similarity:.4f}")
        
        # Sort by similarity
        album_scores.sort(key=lambda x: x[3], reverse=True)
        return album_scores
    
    def process_albums_parallel(self, albums, ref_embeddings, weighted=False, max_workers=4):
        """Process albums in parallel to speed up embedding generation
        
        Args:
            albums (list): List of album tuples
            ref_embeddings (list): List of reference embeddings or weighted embeddings
            weighted (bool): Whether to use weighted similarity calculation
            max_workers (int): Maximum number of worker threads
            
        Returns:
            list: List of (album_id, artist, title, score) tuples
        """
        def process_album(album):
            album_id, artist, title, track_titles = album
            
            # Create album description
            description = f"Artist: {artist}. Album: {title}. Tracks: {track_titles}"
            
            # Get embedding
            album_embedding = self.get_embedding(description)
            if not album_embedding:
                self.logger.warning(f"Skipping album: {artist} - {title} (failed to get embedding)")
                return None
                
            # Calculate similarity
            if weighted:
                similarity, _ = self.calculate_similarity_weighted(album_embedding, ref_embeddings)
            else:
                similarity, _ = self.calculate_similarity(album_embedding, ref_embeddings)
                
            return (album_id, artist, title, similarity)
        
        # Process albums in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_album, albums))
        
        # Filter out None results
        album_scores = [result for result in results if result is not None]
        
        # Sort by similarity
        album_scores.sort(key=lambda x: x[3], reverse=True)
        return album_scores
    
    def create_playlist(self, name=None, limit=50, method='combined'):
        """Create a new playlist with top-ranked albums
        
        Args:
            name (str, optional): Name for the playlist
            limit (int): Maximum number of albums to include
            method (str): Method to use for ranking
            
        Returns:
            bool: True if playlist was created successfully, False otherwise
        """
        # Generate playlist name if not provided
        if not name:
            if self.profile_name:
                profile_display = self.profile_name.replace('_', ' ').title()
                name = f"{profile_display} Mix"
            else:
                name = "Jazz-Electronic-Lounge Mix"
        
        # Rank albums
        ranked_albums = self.rank_albums(method=method)
        if not ranked_albums:
            self.logger.error("No ranked albums available")
            return False
            
        # Take top albums up to limit
        top_albums = ranked_albums[:limit]
        self.logger.info(f"Creating playlist with top {len(top_albums)} albums")
        
        # Connect to database
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            # Check if playlist with this name already exists
            cursor.execute("SELECT id FROM playlists WHERE name = ?", (name,))
            existing = cursor.fetchone()
            
            if existing:
                self.logger.warning(f"Playlist '{name}' already exists. Updating...")
                playlist_id = existing[0]
                # Remove existing tracks
                cursor.execute("DELETE FROM playlist_tracks WHERE playlist_id = ?", (playlist_id,))
                # Update modification date
                cursor.execute(
                    "UPDATE playlists SET modified_date = CURRENT_TIMESTAMP WHERE id = ?", 
                    (playlist_id,)
                )
            else:
                # Create new playlist
                cursor.execute(
                    """
                    INSERT INTO playlists (name, created_date, modified_date) 
                    VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """, 
                    (name,)
                )
                playlist_id = cursor.lastrowid
            
            # Get tracks from top albums
            album_ids = [album_id for album_id, _, _, _ in top_albums]
            placeholders = ','.join(['?'] * len(album_ids))
            
            cursor.execute(
                f"""
                SELECT id FROM tracks 
                WHERE album_id IN ({placeholders})
                ORDER BY album_id, disc_number, track_number
                """, 
                album_ids
            )
            
            tracks = cursor.fetchall()
            
            # Add tracks to playlist
            for position, (track_id,) in enumerate(tracks, 1):
                cursor.execute(
                    """
                    INSERT INTO playlist_tracks (playlist_id, track_id, position, date_added)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """, 
                    (playlist_id, track_id, position)
                )
            
            conn.commit()
            self.logger.info(f"Created playlist '{name}' with {len(tracks)} tracks from {len(top_albums)} albums")
            
            # Log top 10 albums for reference
            self.logger.info("Top 10 albums in playlist:")
            for i, (_, artist, title, score) in enumerate(top_albums[:10]):
                self.logger.info(f"{i+1}. {artist} - {title} (Score: {score:.4f})")
                
            return True
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error creating playlist: {str(e)}")
            return False
            
        finally:
            conn.close()
            
    def export_rankings(self, filename=None, limit=None, method='combined'):
        """Export album rankings to a CSV file
        
        Args:
            filename (str, optional): Path to output file
            limit (int, optional): Maximum number of albums to include
            method (str): Method to use for ranking
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        import csv
        
        # Generate filename if not provided
        if not filename:
            if self.profile_name:
                profile_part = self.profile_name.replace('_', '-')
            else:
                profile_part = "jazz-electronic-lounge"
                
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{profile_part}-rankings-{timestamp}.csv"
            
            # If cache directory exists, use it
            if self.cache_dir:
                filename = os.path.join(self.cache_dir, filename)
        
        # Rank albums
        ranked_albums = self.rank_albums(method=method)
        if not ranked_albums:
            self.logger.error("No ranked albums available")
            return False
            
        # Apply limit if specified
        if limit is not None:
            ranked_albums = ranked_albums[:limit]
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['Rank', 'Album ID', 'Artist', 'Album', 'Score'])
                
                # Write data
                for i, (album_id, artist, title, score) in enumerate(ranked_albums, 1):
                    writer.writerow([i, album_id, artist, title, f"{score:.6f}"])
            
            self.logger.info(f"Exported {len(ranked_albums)} album rankings to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting rankings: {str(e)}")
            return False
    
    def get_album_details(self, album_id):
        """Get detailed information about an album
        
        Args:
            album_id (int): ID of the album
            
        Returns:
            dict: Album details
        """
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            # Get album info
            cursor.execute("""
                SELECT 
                    albums.id,
                    albums.title,
                    artists.name AS artist,
                    albums.format,
                    albums.first_seen,
                    albums.last_seen,
                    albums.play_count
                FROM 
                    albums
                JOIN 
                    artists ON albums.artist_id = artists.id
                WHERE 
                    albums.id = ?
            """, (album_id,))
            
            album_row = cursor.fetchone()
            if not album_row:
                return None
                
            album_id, title, artist, format, first_seen, last_seen, play_count = album_row
            
            # Get tracks
            cursor.execute("""
                SELECT 
                    id,
                    title,
                    track_number,
                    disc_number,
                    duration,
                    play_count
                FROM 
                    tracks
                WHERE 
                    album_id = ?
                ORDER BY 
                    disc_number, track_number
            """, (album_id,))
            
            tracks = []
            for track_row in cursor.fetchall():
                track_id, track_title, track_number, disc_number, duration, track_play_count = track_row
                tracks.append({
                    'id': track_id,
                    'title': track_title,
                    'track_number': track_number,
                    'disc_number': disc_number,
                    'duration': duration,
                    'play_count': track_play_count
                })
            
            # Create album details
            album_details = {
                'id': album_id,
                'title': title,
                'artist': artist,
                'format': format,
                'first_seen': first_seen,
                'last_seen': last_seen,
                'play_count': play_count,
                'tracks': tracks,
                'track_count': len(tracks)
            }
            
            return album_details
            
        except sqlite3.Error as e:
            self.logger.error(f"Error getting album details: {str(e)}")
            return None
        finally:
            conn.close()
    
    def analyze_recommendations(self, top_n=10, method='combined'):
        """Analyze the top recommendations to understand what's being recommended
        
        Args:
            top_n (int): Number of top albums to analyze
            method (str): Method to use for ranking
            
        Returns:
            dict: Analysis results
        """
        # Rank albums
        ranked_albums = self.rank_albums(method=method)
        if not ranked_albums:
            self.logger.error("No ranked albums available")
            return None
            
        # Take top albums
        top_albums = ranked_albums[:top_n]
        
        # Get detailed information for each album
        detailed_albums = []
        for album_id, artist, title, score in top_albums:
            album_details = self.get_album_details(album_id)
            if album_details:
                album_details['score'] = score
                detailed_albums.append(album_details)
        
        # Calculate statistics
        total_tracks = sum(album['track_count'] for album in detailed_albums)
        avg_tracks_per_album = total_tracks / len(detailed_albums) if detailed_albums else 0
        
        total_play_count = sum(album['play_count'] or 0 for album in detailed_albums)
        avg_play_count = total_play_count / len(detailed_albums) if detailed_albums else 0
        
        # Prepare summary
        analysis = {
            'method': method,
            'top_albums': detailed_albums,
            'stats': {
                'total_albums': len(detailed_albums),
                'total_tracks': total_tracks,
                'avg_tracks_per_album': avg_tracks_per_album,
                'total_play_count': total_play_count,
                'avg_play_count': avg_play_count
            }
        }
        
        # Log summary
        self.logger.info(f"Analysis of top {len(detailed_albums)} recommendations:")
        self.logger.info(f"Total tracks: {total_tracks}")
        self.logger.info(f"Average tracks per album: {avg_tracks_per_album:.2f}")
        self.logger.info(f"Total play count: {total_play_count}")
        self.logger.info(f"Average play count: {avg_play_count:.2f}")
        
        return analysis


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Rank music albums based on preferences')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--log', '-l', type=str,
                       help='Path to log file')
    parser.add_argument('--profile', '-p', type=str,
                       help='Name of preference profile to use')
    parser.add_argument('--method', '-m', type=str, default='combined',
                       choices=['standard', 'artist', 'weighted', 'combined'],
                       help='Method to use for ranking')
    parser.add_argument('--playlist', '-pl', type=str,
                       help='Create playlist with this name')
    parser.add_argument('--limit', '-lm', type=int, default=50,
                       help='Maximum number of albums to include in playlist')
    parser.add_argument('--batch-size', '-b', type=int,
                       help='Process albums in batches of this size')
    parser.add_argument('--export', '-e', type=str,
                       help='Export rankings to this file')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Analyze top recommendations')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Initialize ranker
    ranker = PreferenceRanker(args.config, args.log, args.profile, args.verbose)
    
    # Create playlist if requested
    if args.playlist:
        success = ranker.create_playlist(args.playlist, args.limit, args.method)
        if success:
            print(f"Created playlist: {args.playlist}")
        else:
            print("Failed to create playlist")
    
    # Export rankings if requested
    if args.export:
        success = ranker.export_rankings(args.export, args.limit, args.method)
        if success:
            print(f"Exported rankings to: {args.export}")
        else:
            print("Failed to export rankings")
    
    # Analyze recommendations if requested
    if args.analyze:
        analysis = ranker.analyze_recommendations(args.limit, args.method)
        if analysis:
            print(f"Analyzed top {len(analysis['top_albums'])} recommendations")
        else:
            print("Failed to analyze recommendations")
    
    # If no specific action requested, just rank albums and print top 10
    if not (args.playlist or args.export or args.analyze):
        ranked_albums = ranker.rank_albums(args.method, args.batch_size)
        print(f"Top {min(10, len(ranked_albums))} albums:")
        for i, (album_id, artist, title, score) in enumerate(ranked_albums[:10], 1):
            print(f"{i}. {artist} - {title} (Score: {score:.4f})")