#!/usr/bin/env python3
# music_db_builder.py
# A tool for managing a music library database and playlists for SP3000 devices

import os
import sys
import logging
import argparse
import sqlite3
import yaml
import time
from datetime import datetime
from pathlib import Path
import concurrent.futures
from preference_ranker import PreferenceRanker

# Configure logging
def setup_logging(log_path, verbose=False):
    """Set up logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set up file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger("music_db_builder")

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)

def initialize_database(db_path):
    """Initialize the database if it doesn't exist"""
    # Check if database exists
    db_exists = os.path.exists(db_path)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # Create tables if database doesn't exist
    if not db_exists:
        # Create scan_history table
        cursor.execute("""
        CREATE TABLE scan_history (
            id INTEGER PRIMARY KEY, 
            scan_time TIMESTAMP NOT NULL,
            complete_scan BOOLEAN NOT NULL
        )
        """)
        
        # Create artists table
        cursor.execute("""
        CREATE TABLE artists (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            sort_name TEXT,
            first_seen TIMESTAMP NOT NULL,
            last_seen TIMESTAMP NOT NULL
        )
        """)
        
        # Create albums table
        cursor.execute("""
        CREATE TABLE albums (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            artist TEXT NOT NULL,
            title TEXT NOT NULL,
            format TEXT NOT NULL,
            boxset TEXT,
            mtime TIMESTAMP NOT NULL,
            first_seen TIMESTAMP NOT NULL,
            last_seen TIMESTAMP NOT NULL,
            play_count INTEGER DEFAULT 0,
            artist_id INTEGER REFERENCES artists(id),
            artwork BLOB,
            artwork_mime_type TEXT
        )
        """)
        
        # Create tracks table
        cursor.execute("""
        CREATE TABLE tracks (
            id INTEGER PRIMARY KEY,
            album_id INTEGER NOT NULL,
            path TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            track_number INTEGER,
            disc_number INTEGER,
            duration REAL,
            mtime TIMESTAMP NOT NULL,
            first_seen TIMESTAMP NOT NULL,
            last_seen TIMESTAMP NOT NULL,
            play_count INTEGER DEFAULT 0,
            artist_id INTEGER REFERENCES artists(id),
            artist TEXT,
            FOREIGN KEY (album_id) REFERENCES albums (id) ON DELETE CASCADE
        )
        """)
        
        # Create playlists table
        cursor.execute("""
        CREATE TABLE playlists (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            created_date TIMESTAMP NOT NULL,
            modified_date TIMESTAMP NOT NULL
        )
        """)
        
        # Create playlist_tracks table
        cursor.execute("""
        CREATE TABLE playlist_tracks (
            playlist_id INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            date_added TIMESTAMP NOT NULL,
            PRIMARY KEY (playlist_id, track_id),
            FOREIGN KEY (playlist_id) REFERENCES playlists (id) ON DELETE CASCADE,
            FOREIGN KEY (track_id) REFERENCES tracks (id) ON DELETE CASCADE
        )
        """)
        
        # Create sync_history table
        cursor.execute("""
        CREATE TABLE sync_history (
            id INTEGER PRIMARY KEY,
            track_id INTEGER NOT NULL,
            sync_date TIMESTAMP NOT NULL,
            device TEXT NOT NULL,
            FOREIGN KEY (track_id) REFERENCES tracks (id) ON DELETE CASCADE
        )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX idx_albums_path ON albums (path)")
        cursor.execute("CREATE INDEX idx_tracks_path ON tracks (path)")
        cursor.execute("CREATE INDEX idx_tracks_album_id ON tracks (album_id)")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()

def scan_library(config, rebuild=False, clear_cache=False):
    """Scan music library and update database"""
    logger = logging.getLogger("music_db_builder")
    logger.info("Starting library scan")
    
    # Get paths from config
    base_path = config['paths']['base']
    db_path = config['paths']['database']
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear database if rebuilding
    if rebuild:
        logger.info("Rebuilding database from scratch")
        cursor.execute("DELETE FROM scan_history")
        cursor.execute("DELETE FROM sync_history")
        cursor.execute("DELETE FROM playlist_tracks")
        cursor.execute("DELETE FROM playlists")
        cursor.execute("DELETE FROM tracks")
        cursor.execute("DELETE FROM albums")
        cursor.execute("DELETE FROM artists")
        conn.commit()
    
    # Start scan
    scan_time = datetime.now()
    
    # Insert scan record
    cursor.execute(
        "INSERT INTO scan_history (scan_time, complete_scan) VALUES (?, ?)",
        (scan_time, rebuild)
    )
    scan_id = cursor.lastrowid
    conn.commit()
    
    # Scan directories
    logger.info(f"Scanning directory: {base_path}")
    
    # TODO: Implement actual scanning logic here
    # This is just a placeholder
    
    # Update scan record
    cursor.execute(
        "UPDATE scan_history SET complete_scan = ? WHERE id = ?",
        (True, scan_id)
    )
    conn.commit()
    
    # Close database connection
    conn.close()
    
    logger.info("Library scan completed")

def process_playlists(config):
    """Process playlists from source to export directory"""
    logger = logging.getLogger("music_db_builder")
    logger.info("Processing playlists")
    
    # Get paths from config
    db_path = config['paths']['database']
    playlists_source = config['paths']['playlists_source']
    playlists_export = config['paths']['playlists_export']
    
    # Create export directory if it doesn't exist
    os.makedirs(playlists_export, exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # TODO: Implement playlist processing logic
    # This is just a placeholder
    
    # Close database connection
    conn.close()
    
    logger.info("Playlist processing completed")

def scan_artists(config):
    """Refresh artist information from tracks"""
    logger = logging.getLogger("music_db_builder")
    logger.info("Scanning artists")
    
    # Get database path from config
    db_path = config['paths']['database']
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # TODO: Implement artist scanning logic
    # This is just a placeholder
    
    # Close database connection
    conn.close()
    
    logger.info("Artist scanning completed")

def add_preference_ranking_arguments(parser):
    """Add preference ranking arguments to the parser"""
    ranking_group = parser.add_argument_group('Preference Ranking')
    ranking_group.add_argument('--rank-preferences', '-rp', action='store_true',
                        help='Rank albums by preferences and create playlist')
    ranking_group.add_argument('--profile', '-pr', type=str,
                        help='Name of the preference profile to use (from config)')
    ranking_group.add_argument('--ranking-method', '-rm', type=str, default='combined',
                        choices=['standard', 'artist', 'weighted', 'combined'],
                        help='Method to use for ranking (default: combined)')
    ranking_group.add_argument('--playlist-name', '-pn', type=str, default=None,
                        help='Name for the generated playlist (default: based on profile)')
    ranking_group.add_argument('--limit', '-l', type=int, default=50,
                        help='Maximum number of albums to include in playlist')
    ranking_group.add_argument('--export-rankings', '-er', action='store_true',
                        help='Export rankings to CSV file')
    ranking_group.add_argument('--analyze-rankings', '-ar', action='store_true',
                        help='Analyze top ranked albums')

def run_preference_ranking(config_path, log_path, args):
    """Run preference ranking operations"""
    logger = logging.getLogger("music_db_builder")
    
    try:
        logger.info("Initializing preference ranker...")
        ranker = PreferenceRanker(
            config_path=config_path,
            log_path=log_path,
            profile_name=args.profile,
            verbose=args.verbose
        )
        
        # Determine playlist name if not specified
        playlist_name = args.playlist_name
        if not playlist_name:
            profile_name = args.profile or ranker.profile_name or "Default"
            playlist_name = f"{profile_name.replace('_', ' ').title()} Mix"
        
        # Create playlist
        logger.info(f"Creating playlist '{playlist_name}' using {args.ranking_method} method...")
        success = ranker.create_playlist(
            name=playlist_name,
            limit=args.limit,
            method=args.ranking_method
        )
        
        if success:
            logger.info(f"Successfully created playlist: {playlist_name}")
        else:
            logger.error("Failed to create preference-based playlist")
            return False
        
        # Export rankings if requested
        if args.export_rankings:
            logger.info("Exporting rankings to CSV...")
            export_path = os.path.join(
                os.path.dirname(log_path), 
                f"{playlist_name.replace(' ', '_').lower()}_rankings.csv"
            )
            success = ranker.export_rankings(export_path, args.limit, args.ranking_method)
            if success:
                logger.info(f"Rankings exported to: {export_path}")
            else:
                logger.error("Failed to export rankings")
        
        # Analyze rankings if requested
        if args.analyze_rankings:
            logger.info("Analyzing top ranked albums...")
            analysis = ranker.analyze_recommendations(args.limit, args.ranking_method)
            if analysis:
                logger.info(f"Analysis complete for top {len(analysis['top_albums'])} albums")
            else:
                logger.error("Failed to analyze rankings")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during preference ranking: {str(e)}", exc_info=True)
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Music Library Manager for SP3000')
    
    parser.add_argument('--config', '-c', type=str, 
                        default=os.path.expanduser('~/sp3000sync/config.yaml'),
                        help='Path to config file')
    parser.add_argument('--rebuild', '-r', action='store_true',
                        help='Rebuild database from scratch')
    parser.add_argument('--init-config', '-i', action='store_true',
                        help='Create default config file if none exists')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear cache before scanning')
    parser.add_argument('--process-playlists', '-p', action='store_true',
                        help='Process playlists')
    parser.add_argument('--scan-artists', action='store_true',
                        help='Refresh artist information from tracks')
    
    # Add preference ranking arguments
    add_preference_ranking_arguments(parser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize configuration
    config_path = args.config
    
    # Create default config if requested
    if args.init_config and not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
    default_config = {
        'paths': {
            'base': os.path.expanduser('~/Music'),
            'database': os.path.expanduser('~/sp3000sync/music_library.db'),
            'log': os.path.expanduser('~/sp3000sync/music_library.log'),
            'cache': os.path.expanduser('~/sp3000sync/cache'),
            'playlists_source': os.path.expanduser('~/Music/Playlists'),
            'playlists_export': os.path.expanduser('~/Music/Playlists_sp3000'),
            'device_root': '/storage/Music/'
        },
        'formats': {
            'audio': ['.flac', '.dsf', '.wav', '.mp3']
        },
        'scan': {
            'threads': 4,
            'batch_size': 100,
            'use_cache': True
        },
        'llm': {
            'api_key': '',
            'api_url': 'https://api.anthropic.com/v1/messages',
            'embedding_url': 'https://api.anthropic.com/v1/embeddings',
            'model': 'claude-3-sonnet-20240229',
            'embedding_model': 'claude-3-sonnet-20240229-embedding'
        },
        'music_preferences': {
            'default_profile': 'jazz_electronic_funk',
            'profiles': {
                'jazz_electronic_funk': {
                    # Standard weight references (all equal importance)
                    'reference_points': [
                        "Jazz fusion with electronic elements and smooth textures",
                        "Downtempo electronic music with jazz instrumentation",
                        "Nu-jazz with atmospheric electronic production",
                        "Lounge music with jazz samples and electronic beats",
                        "Acid jazz with electronic synthesizers and dance elements",
                        "Chill-out electronic music with jazz influences",
                        "Electronic lounge music with improvisational jazz solos",
                        "Soulpersona-style modern soul with vintage production techniques and smooth R&B influences",
                        "Brand New Heavies-inspired acid jazz with funk basslines, horn sections and soulful vocals",
                        "Jamiroquai-style funk with disco elements, synth textures and infectious grooves",
                        "Neo-soul with jazz-funk instrumentation and sophisticated chord progressions",
                        "Classic rare groove and R&B with modern production values and sampling techniques",
                        "70s-inspired funk with tight horn sections, Rhodes piano and syncopated rhythms",
                        "Deep house music with soulful vocals and jazz-influenced keyboard solos",
                        "Soul and funk with broken beat electronic production and jazzy arrangements",
                        "Contemporary jazz-funk with hip-hop influences and sophisticated improvisations",
                        "Sophisticated R&B with jazz harmony and deep groove sensibilities",
                        "Instrumental soul-jazz with funky drum breaks and melodic horn arrangements",
                        "Urban jazz with soul vocals and contemporary electronic production",
                        "Incognito-style jazz-funk fusion with smooth production and soulful ensemble playing",
                        "Soul II Soul-inspired R&B with electronic beats and atmospheric string arrangements",
                        "Jazz-house fusion with deep grooves and sophisticated instrumental solos",
                        "Downtempo funk with atmospheric production and vintage keyboard sounds",
                        "Kyoto Jazz Massive-style nu-jazz with global influences and electronic textures",
                        "Hyperdrive Sound-inspired broken beat with complex rhythms and soulful elements",
                        "Luna Lounge-style atmospheric jazz with electronic underpinnings and ambient textures",
                        "Kraftwerk-inspired electronic minimalism with jazz harmonies and funk grooves"
                    ],
                    
                    # Artist-specific references
                    'artist_references': [
                        "Music similar to Soulpersona's modern soul with vintage production techniques",
                        "Albums that sound like Brand New Heavies with their acid jazz, funk basslines and horn sections",
                        "Tracks with Jamiroquai's style of funk, disco elements and infectious grooves",
                        "Music in the style of Incognito's sophisticated jazz-funk fusion",
                        "Albums with the Soul II Soul aesthetic of R&B with electronic beats and string arrangements",
                        "Tracks that feature Kyoto Jazz Massive's nu-jazz style with global influences",
                        "Music similar to Hyperdrive Sound's broken beat with complex rhythms",
                        "Albums with the Luna Lounge vibe of atmospheric jazz and lounge elements",
                        "Tracks influenced by Kraftwerk's electronic minimalism but with jazz harmonies",
                        "Music that combines Carl Hudson's keyboard jazz with electronic elements",
                        "Albums similar to Miles Davis' electric period with funk and electronic elements",
                        "Tracks that capture the spirit of Donald Byrd's jazz-funk fusion era",
                        "Music with Freddie Hubbard's sophisticated jazz trumpet with modern production",
                        "Albums similar to James Taylor Quartet's acid jazz and funk instrumentals"
                    ],
                    
                    # Weighted references (higher weight = stronger preference)
                    'weighted_references': [
                        {'reference': "Soulpersona-style modern soul with vintage production techniques and smooth R&B influences", 'weight': 3.0},
                        {'reference': "Brand New Heavies-inspired acid jazz with funk basslines, horn sections and soulful vocals", 'weight': 3.0},
                        {'reference': "Jamiroquai-style funk with disco elements, synth textures and infectious grooves", 'weight': 3.0},
                        {'reference': "Neo-soul with jazz-funk instrumentation and sophisticated chord progressions", 'weight': 2.5},
                        {'reference': "Acid jazz with electronic synthesizers and dance elements", 'weight': 2.5},
                        {'reference': "Jazz fusion with electronic elements and smooth textures", 'weight': 2.0},
                        {'reference': "Incognito-style jazz-funk fusion with smooth production and soulful ensemble playing", 'weight': 2.0},
                        {'reference': "Instrumental soul-jazz with funky drum breaks and melodic horn arrangements", 'weight': 2.0},
                        {'reference': "Kyoto Jazz Massive-style nu-jazz with global influences and electronic textures", 'weight': 2.0},
                        {'reference': "Soul and funk with broken beat electronic production and jazzy arrangements", 'weight': 1.5},
                        {'reference': "Luna Lounge-style atmospheric jazz with electronic underpinnings and ambient textures", 'weight': 1.5},
                        {'reference': "Lounge music with jazz samples and electronic beats", 'weight': 1.5},
                        {'reference': "Classic rare groove and R&B with modern production values and sampling techniques", 'weight': 1.5},
                        {'reference': "Deep house music with soulful vocals and jazz-influenced keyboard solos", 'weight': 1.0},
                        {'reference': "Kraftwerk-inspired electronic minimalism with jazz harmonies and funk grooves", 'weight': 1.0},
                        {'reference': "Downtempo electronic music with jazz instrumentation", 'weight': 1.0},
                        {'reference': "Electronic lounge music with improvisational jazz solos", 'weight': 1.0},
                        {'reference': "Chill-out electronic music with jazz influences", 'weight': 1.0}
                    ]
                }
            }
        }
    }
        
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
        
    print(f"Created default config file at {config_path}")
    sys.exit(0)
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    log_path = config['paths']['log']
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logging(log_path, args.verbose)
    
    # Initialize database
    db_path = config['paths']['database']
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    initialize_database(db_path)
    
    # Process command line options
    operations_performed = False
    
    # Scan library if no specific operations requested or if rebuild requested
    if args.rebuild or (not args.process_playlists and not args.scan_artists and not args.rank_preferences):
        scan_library(config, args.rebuild, args.clear_cache)
        operations_performed = True
    
    # Process playlists if requested
    if args.process_playlists:
        process_playlists(config)
        operations_performed = True
    
    # Scan artists if requested
    if args.scan_artists:
        scan_artists(config)
        operations_performed = True
    
    # Rank preferences if requested
    if args.rank_preferences:
        logger.info("Ranking albums by preferences...")
        success = run_preference_ranking(config_path, log_path, args)
        if not success:
            logger.error("Preference ranking failed")
        operations_performed = True
    
    # If no operations performed, print help
    if not operations_performed:
        parser.print_help()
    
    logger.info("All operations completed")

if __name__ == "__main__":
    main()