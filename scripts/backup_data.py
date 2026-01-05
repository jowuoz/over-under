#!/usr/bin/env python3
"""
backup_data.py - Comprehensive backup system for Over/Under Predictor
Handles data backup, restoration, and management with encryption and compression
"""
import os
import sys
import json
import yaml
import zipfile
import tarfile
import hashlib
import shutil
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import asyncio
import aiofiles
import aioboto3  # For S3 backups (optional)
import gzip
import bz2
import lzma
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import argparse
import subprocess
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try imports with fallbacks
try:
    from config import get_config, load_config
except ImportError:
    print("Warning: Could not import config module")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backup")


class BackupManager:
    """
    Comprehensive backup manager for Over/Under Predictor system
    Supports local, cloud, and encrypted backups
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize backup manager
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            try:
                self.config = get_config()
            except:
                self.config = self._load_default_config()
        
        # Setup paths
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.storage_dir = self.project_root / 'storage'
        self.backup_dir = self.storage_dir / 'backups'
        self.data_dir = self.storage_dir / 'data'
        self.log_dir = self.storage_dir / 'logs'
        self.reports_dir = self.project_root / 'reports'
        self.config_dir = self.project_root / 'config'
        
        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup configuration
        self.backup_config = {
            'retention_days': self.config.get('backup', {}).get('retention_days', 30),
            'max_backups': self.config.get('backup', {}).get('max_backups', 100),
            'compression': self.config.get('backup', {}).get('compression', 'gzip'),
            'encryption': self.config.get('backup', {}).get('encryption', False),
            'encryption_key': self.config.get('backup', {}).get('encryption_key', ''),
            'schedule': self.config.get('backup', {}).get('schedule', 'daily'),
            'cloud_backup': self.config.get('backup', {}).get('cloud_backup', False),
            'backup_types': ['full', 'incremental', 'differential']
        }
        
        # Cloud configuration (optional)
        self.cloud_config = {
            's3_bucket': os.getenv('AWS_S3_BUCKET', ''),
            's3_prefix': os.getenv('AWS_S3_PREFIX', 'over-under-predictor/backups/'),
            'aws_access_key': os.getenv('AWS_ACCESS_KEY_ID', ''),
            'aws_secret_key': os.getenv('AWS_SECRET_ACCESS_KEY', ''),
            'aws_region': os.getenv('AWS_REGION', 'us-east-1')
        }
        
        # Database configuration
        self.db_config = {
            'predictions_file': self.data_dir / 'predictions.json',
            'historical_dir': self.data_dir / 'historical',
            'models_dir': self.storage_dir / 'models',
            'cache_dir': self.storage_dir / 'cache'
        }
        
        # Backup manifest
        self.manifest = {
            'version': '1.0.0',
            'project': 'Over/Under Predictor',
            'backup_system': 'BackupManager v1.0'
        }
        
        # Statistics
        self.stats = {
            'total_backups': 0,
            'total_size_gb': 0,
            'last_backup': None,
            'successful_backups': 0,
            'failed_backups': 0
        }
        
        # Initialize encryption if enabled
        if self.backup_config['encryption'] and self.backup_config['encryption_key']:
            self.cipher = self._init_encryption()
        else:
            self.cipher = None
        
        logger.info(f"Backup manager initialized for project: {self.project_root}")
        logger.info(f"Backup directory: {self.backup_dir}")
        logger.info(f"Retention: {self.backup_config['retention_days']} days")
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'backup': {
                'retention_days': 30,
                'max_backups': 100,
                'compression': 'gzip',
                'encryption': False,
                'schedule': 'daily',
                'cloud_backup': False
            }
        }
    
    def _init_encryption(self) -> Optional[Fernet]:
        """Initialize encryption cipher"""
        try:
            key = self.backup_config['encryption_key'].encode()
            
            # If key is not 32 bytes, derive a key
            if len(key) != 32:
                salt = b'over_under_predictor_salt'
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(key))
            else:
                key = base64.urlsafe_b64encode(key)
            
            return Fernet(key)
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            return None
    
    async def create_backup(self, backup_type: str = 'incremental',
                          description: str = '') -> Dict:
        """
        Create a new backup
        
        Args:
            backup_type: Type of backup (full, incremental, differential)
            description: Optional description for the backup
            
        Returns:
            Dictionary with backup information
        """
        logger.info(f"Creating {backup_type} backup...")
        
        timestamp = datetime.now()
        backup_id = self._generate_backup_id(timestamp, backup_type)
        backup_path = self.backup_dir / backup_id
        
        try:
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Create manifest
            manifest = self._create_manifest(timestamp, backup_type, description)
            
            # Perform backup based on type
            if backup_type == 'full':
                backup_result = await self._create_full_backup(backup_path, manifest)
            elif backup_type == 'incremental':
                backup_result = await self._create_incremental_backup(backup_path, manifest)
            elif backup_type == 'differential':
                backup_result = await self._create_differential_backup(backup_path, manifest)
            else:
                raise ValueError(f"Unknown backup type: {backup_type}")
            
            # Compress backup
            if self.backup_config['compression']:
                compressed_path = await self._compress_backup(backup_path, backup_id)
                backup_result['compressed_file'] = str(compressed_path)
                backup_result['compression'] = self.backup_config['compression']
                
                # Remove uncompressed directory
                shutil.rmtree(backup_path)
                backup_path = compressed_path
            
            # Encrypt backup if enabled
            if self.cipher:
                encrypted_path = await self._encrypt_backup(backup_path)
                backup_result['encrypted_file'] = str(encrypted_path)
                backup_result['encryption'] = True
                
                # Remove unencrypted file
                os.remove(backup_path)
                backup_path = encrypted_path
            
            # Update manifest with final information
            manifest.update({
                'backup_size': os.path.getsize(backup_path),
                'backup_format': 'compressed' if self.backup_config['compression'] else 'directory',
                'encrypted': self.cipher is not None,
                'final_path': str(backup_path)
            })
            
            # Save manifest
            manifest_path = self.backup_dir / f"{backup_id}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Update statistics
            self._update_stats(backup_result, True)
            
            # Cloud backup if enabled
            if self.backup_config['cloud_backup']:
                cloud_result = await self._upload_to_cloud(backup_path, manifest_path, backup_id)
                backup_result['cloud_backup'] = cloud_result
            
            logger.info(f"Backup created successfully: {backup_id}")
            logger.info(f"  Size: {self._format_size(backup_result.get('total_size', 0))}")
            logger.info(f"  Files: {backup_result.get('file_count', 0)}")
            logger.info(f"  Path: {backup_path}")
            
            return {
                'status': 'success',
                'backup_id': backup_id,
                'backup_type': backup_type,
                'timestamp': timestamp.isoformat(),
                'path': str(backup_path),
                'manifest_path': str(manifest_path),
                'details': backup_result
            }
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            
            # Cleanup on failure
            if os.path.exists(backup_path):
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
                else:
                    os.remove(backup_path)
            
            self._update_stats({}, False)
            
            return {
                'status': 'failed',
                'backup_id': backup_id,
                'error': str(e),
                'timestamp': timestamp.isoformat()
            }
    
    def _generate_backup_id(self, timestamp: datetime, backup_type: str) -> str:
        """Generate unique backup ID"""
        date_str = timestamp.strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(str(timestamp.timestamp()).encode()).hexdigest()[:8]
        return f"{date_str}_{backup_type}_{random_suffix}"
    
    def _create_manifest(self, timestamp: datetime, backup_type: str,
                        description: str) -> Dict:
        """Create backup manifest"""
        return {
            **self.manifest,
            'backup_id': self._generate_backup_id(timestamp, backup_type),
            'backup_type': backup_type,
            'timestamp': timestamp.isoformat(),
            'description': description,
            'system_info': self._get_system_info(),
            'config_hash': self._get_config_hash(),
            'file_list': [],
            'checksums': {}
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        import platform
        
        return {
            'python_version': platform.python_version(),
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'hostname': platform.node(),
            'username': os.getenv('USER', os.getenv('USERNAME', 'unknown'))
        }
    
    def _get_config_hash(self) -> str:
        """Calculate hash of configuration"""
        config_files = [
            self.project_root / 'config' / 'settings.yaml',
            self.project_root / 'requirements.txt',
            self.project_root / '.github' / 'workflows' / 'predictor.yml'
        ]
        
        hasher = hashlib.sha256()
        for config_file in config_files:
            if config_file.exists():
                hasher.update(config_file.read_bytes())
        
        return hasher.hexdigest()
    
    async def _create_full_backup(self, backup_path: Path,
                                 manifest: Dict) -> Dict:
        """
        Create full backup of all data
        
        Args:
            backup_path: Path to backup directory
            manifest: Backup manifest
            
        Returns:
            Dictionary with backup details
        """
        logger.info("Creating full backup...")
        
        backup_details = {
            'type': 'full',
            'file_count': 0,
            'total_size': 0,
            'files': {}
        }
        
        # Backup data directory
        data_backup = backup_path / 'data'
        data_backup.mkdir(parents=True, exist_ok=True)
        
        if self.data_dir.exists():
            data_files = await self._backup_directory(self.data_dir, data_backup)
            backup_details['files']['data'] = data_files
            backup_details['file_count'] += data_files['file_count']
            backup_details['total_size'] += data_files['total_size']
            
            # Add to manifest
            manifest['file_list'].extend(data_files['file_list'])
            manifest['checksums'].update(data_files['checksums'])
        
        # Backup logs directory (optional)
        logs_backup = backup_path / 'logs'
        logs_backup.mkdir(parents=True, exist_ok=True)
        
        if self.log_dir.exists():
            # Only backup recent logs (last 7 days)
            recent_logs = await self._get_recent_files(self.log_dir, days=7)
            if recent_logs:
                logs_files = await self._backup_files(recent_logs, logs_backup)
                backup_details['files']['logs'] = logs_files
                backup_details['file_count'] += logs_files['file_count']
                backup_details['total_size'] += logs_files['total_size']
                
                manifest['file_list'].extend(logs_files['file_list'])
                manifest['checksums'].update(logs_files['checksums'])
        
        # Backup reports directory
        reports_backup = backup_path / 'reports'
        reports_backup.mkdir(parents=True, exist_ok=True)
        
        if self.reports_dir.exists():
            reports_files = await self._backup_directory(self.reports_dir, reports_backup)
            backup_details['files']['reports'] = reports_files
            backup_details['file_count'] += reports_files['file_count']
            backup_details['total_size'] += reports_files['total_size']
            
            manifest['file_list'].extend(reports_files['file_list'])
            manifest['checksums'].update(reports_files['checksums'])
        
        # Backup configuration
        config_backup = backup_path / 'config'
        config_backup.mkdir(parents=True, exist_ok=True)
        
        if self.config_dir.exists():
            config_files = await self._backup_directory(self.config_dir, config_backup)
            backup_details['files']['config'] = config_files
            backup_details['file_count'] += config_files['file_count']
            backup_details['total_size'] += config_files['total_size']
            
            manifest['file_list'].extend(config_files['file_list'])
            manifest['checksums'].update(config_files['checksums'])
        
        # Backup GitHub workflows
        workflows_dir = self.project_root / '.github' / 'workflows'
        if workflows_dir.exists():
            workflows_backup = backup_path / 'github_workflows'
            workflows_backup.mkdir(parents=True, exist_ok=True)
            
            workflows_files = await self._backup_directory(workflows_dir, workflows_backup)
            backup_details['files']['workflows'] = workflows_files
            backup_details['file_count'] += workflows_files['file_count']
            backup_details['total_size'] += workflows_files['total_size']
            
            manifest['file_list'].extend(workflows_files['file_list'])
            manifest['checksums'].update(workflows_files['checksums'])
        
        # Backup source code (optional)
        src_dir = self.project_root / 'src'
        if src_dir.exists():
            src_backup = backup_path / 'src'
            src_backup.mkdir(parents=True, exist_ok=True)
            
            # Only backup .py files
            py_files = list(src_dir.rglob('*.py'))
            if py_files:
                src_files = await self._backup_files(py_files, src_backup)
                backup_details['files']['src'] = src_files
                backup_details['file_count'] += src_files['file_count']
                backup_details['total_size'] += src_files['total_size']
                
                manifest['file_list'].extend(src_files['file_list'])
                manifest['checksums'].update(src_files['checksums'])
        
        # Create database dump
        db_dump = await self._dump_database()
        if db_dump:
            db_path = backup_path / 'database_dump.json'
            async with aiofiles.open(db_path, 'w') as f:
                await f.write(json.dumps(db_dump, indent=2))
            
            backup_details['database_dump'] = {
                'file': str(db_path),
                'size': os.path.getsize(db_path),
                'tables': len(db_dump)
            }
            backup_details['file_count'] += 1
            backup_details['total_size'] += os.path.getsize(db_path)
            
            manifest['file_list'].append(str(db_path))
            manifest['checksums'][str(db_path)] = self._calculate_checksum(db_path)
        
        logger.info(f"Full backup completed: {backup_details['file_count']} files, "
                   f"{self._format_size(backup_details['total_size'])}")
        
        return backup_details
    
    async def _create_incremental_backup(self, backup_path: Path,
                                        manifest: Dict) -> Dict:
        """
        Create incremental backup (only changed files since last backup)
        
        Args:
            backup_path: Path to backup directory
            manifest: Backup manifest
            
        Returns:
            Dictionary with backup details
        """
        logger.info("Creating incremental backup...")
        
        # Find last backup
        last_backup = self._get_last_backup()
        if not last_backup:
            logger.info("No previous backup found, creating full backup instead")
            return await self._create_full_backup(backup_path, manifest)
        
        # Get last backup timestamp
        last_timestamp = datetime.fromisoformat(last_backup['timestamp'])
        
        backup_details = {
            'type': 'incremental',
            'base_backup': last_backup['backup_id'],
            'file_count': 0,
            'total_size': 0,
            'files': {}
        }
        
        # Find files changed since last backup
        changed_files = await self._get_changed_files_since(last_timestamp)
        
        if not changed_files:
            logger.info("No files changed since last backup")
            return backup_details
        
        # Backup changed files
        files_backup = backup_path / 'changed_files'
        files_backup.mkdir(parents=True, exist_ok=True)
        
        files_result = await self._backup_files(changed_files, files_backup)
        backup_details['files']['changed'] = files_result
        backup_details['file_count'] = files_result['file_count']
        backup_details['total_size'] = files_result['total_size']
        
        # Add to manifest
        manifest['file_list'].extend(files_result['file_list'])
        manifest['checksums'].update(files_result['checksums'])
        manifest['base_backup'] = last_backup['backup_id']
        
        logger.info(f"Incremental backup completed: {backup_details['file_count']} files, "
                   f"{self._format_size(backup_details['total_size'])}")
        
        return backup_details
    
    async def _create_differential_backup(self, backup_path: Path,
                                         manifest: Dict) -> Dict:
        """
        Create differential backup (all changes since last full backup)
        
        Args:
            backup_path: Path to backup directory
            manifest: Backup manifest
            
        Returns:
            Dictionary with backup details
        """
        logger.info("Creating differential backup...")
        
        # Find last full backup
        last_full_backup = self._get_last_backup(backup_type='full')
        if not last_full_backup:
            logger.info("No full backup found, creating full backup instead")
            return await self._create_full_backup(backup_path, manifest)
        
        # Get last full backup timestamp
        last_full_timestamp = datetime.fromisoformat(last_full_backup['timestamp'])
        
        backup_details = {
            'type': 'differential',
            'base_backup': last_full_backup['backup_id'],
            'file_count': 0,
            'total_size': 0,
            'files': {}
        }
        
        # Find all files changed since last full backup
        changed_files = await self._get_changed_files_since(last_full_timestamp)
        
        if not changed_files:
            logger.info("No files changed since last full backup")
            return backup_details
        
        # Backup changed files
        files_backup = backup_path / 'changed_files'
        files_backup.mkdir(parents=True, exist_ok=True)
        
        files_result = await self._backup_files(changed_files, files_backup)
        backup_details['files']['changed'] = files_result
        backup_details['file_count'] = files_result['file_count']
        backup_details['total_size'] = files_result['total_size']
        
        # Add to manifest
        manifest['file_list'].extend(files_result['file_list'])
        manifest['checksums'].update(files_result['checksums'])
        manifest['base_backup'] = last_full_backup['backup_id']
        
        logger.info(f"Differential backup completed: {backup_details['file_count']} files, "
                   f"{self._format_size(backup_details['total_size'])}")
        
        return backup_details
    
    async def _backup_directory(self, source_dir: Path,
                               target_dir: Path) -> Dict:
        """
        Backup entire directory
        
        Args:
            source_dir: Source directory path
            target_dir: Target directory path
            
        Returns:
            Dictionary with backup details
        """
        result = {
            'source': str(source_dir),
            'target': str(target_dir),
            'file_count': 0,
            'total_size': 0,
            'file_list': [],
            'checksums': {}
        }
        
        # Walk through directory
        for root, dirs, files in os.walk(source_dir):
            # Calculate relative path
            rel_path = Path(root).relative_to(source_dir)
            target_path = target_dir / rel_path
            
            # Create target directory
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for file in files:
                source_file = Path(root) / file
                target_file = target_path / file
                
                # Copy file
                shutil.copy2(source_file, target_file)
                
                # Update statistics
                file_size = os.path.getsize(source_file)
                result['file_count'] += 1
                result['total_size'] += file_size
                result['file_list'].append(str(source_file))
                result['checksums'][str(source_file)] = self._calculate_checksum(source_file)
        
        return result
    
    async def _backup_files(self, files: List[Path],
                           target_dir: Path) -> Dict:
        """
        Backup specific files
        
        Args:
            files: List of file paths to backup
            target_dir: Target directory
            
        Returns:
            Dictionary with backup details
        """
        result = {
            'target': str(target_dir),
            'file_count': 0,
            'total_size': 0,
            'file_list': [],
            'checksums': {}
        }
        
        for source_file in files:
            if not source_file.exists():
                continue
            
            # Calculate relative path from project root
            try:
                rel_path = source_file.relative_to(self.project_root)
            except ValueError:
                # If file is outside project root, use filename
                rel_path = Path(source_file.name)
            
            target_file = target_dir / rel_path
            
            # Create target directory if needed
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(source_file, target_file)
            
            # Update statistics
            file_size = os.path.getsize(source_file)
            result['file_count'] += 1
            result['total_size'] += file_size
            result['file_list'].append(str(source_file))
            result['checksums'][str(source_file)] = self._calculate_checksum(source_file)
        
        return result
    
    async def _get_recent_files(self, directory: Path, days: int = 7) -> List[Path]:
        """Get files modified in the last N days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if mtime > cutoff_time:
                    recent_files.append(file_path)
        
        return recent_files
    
    async def _get_changed_files_since(self, since: datetime) -> List[Path]:
        """Get files changed since specified timestamp"""
        changed_files = []
        
        # Directories to check
        check_dirs = [
            self.data_dir,
            self.log_dir,
            self.reports_dir,
            self.config_dir,
            self.project_root / 'src',
            self.project_root / 'scripts'
        ]
        
        for directory in check_dirs:
            if not directory.exists():
                continue
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = Path(root) / file
                    
                    # Skip certain file types
                    if file_path.suffix in ['.pyc', '.tmp', '.log']:
                        continue
                    
                    # Check modification time
                    try:
                        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if mtime > since:
                            changed_files.append(file_path)
                    except OSError:
                        continue
        
        return changed_files
    
    async def _dump_database(self) -> Dict:
        """Dump database content (predictions, historical data)"""
        db_dump = {
            'timestamp': datetime.now().isoformat(),
            'predictions': {},
            'historical': {},
            'statistics': {}
        }
        
        try:
            # Dump predictions file
            if self.db_config['predictions_file'].exists():
                with open(self.db_config['predictions_file'], 'r') as f:
                    predictions_data = json.load(f)
                    db_dump['predictions'] = {
                        'file': str(self.db_config['predictions_file']),
                        'data': predictions_data,
                        'size': os.path.getsize(self.db_config['predictions_file'])
                    }
            
            # Dump historical data
            if self.db_config['historical_dir'].exists():
                historical_files = list(self.db_config['historical_dir'].glob('*.json'))
                db_dump['historical']['file_count'] = len(historical_files)
                db_dump['historical']['total_size'] = sum(
                    os.path.getsize(f) for f in historical_files
                )
                
                # Sample a few recent files
                recent_files = sorted(
                    historical_files,
                    key=lambda f: os.path.getmtime(f),
                    reverse=True
                )[:5]
                
                db_dump['historical']['sample_files'] = [
                    str(f.relative_to(self.project_root)) for f in recent_files
                ]
            
            # Calculate statistics
            total_predictions = 0
            if 'predictions' in db_dump and 'data' in db_dump['predictions']:
                predictions_data = db_dump['predictions']['data']
                total_predictions = len(predictions_data.get('predictions', []))
            
            db_dump['statistics'] = {
                'total_predictions': total_predictions,
                'backup_timestamp': datetime.now().isoformat()
            }
            
            return db_dump
            
        except Exception as e:
            logger.error(f"Failed to dump database: {e}")
            return {}
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        hasher = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return 'error'
    
    async def _compress_backup(self, backup_path: Path,
                              backup_id: str) -> Path:
        """Compress backup directory"""
        compression = self.backup_config['compression']
        output_path = self.backup_dir / f"{backup_id}.tar.{compression}"
        
        logger.info(f"Compressing backup with {compression}...")
        
        if compression == 'gzip':
            with tarfile.open(output_path, 'w:gz') as tar:
                tar.add(backup_path, arcname=backup_path.name)
        
        elif compression == 'bz2':
            with tarfile.open(output_path, 'w:bz2') as tar:
                tar.add(backup_path, arcname=backup_path.name)
        
        elif compression == 'xz':
            with tarfile.open(output_path, 'w:xz') as tar:
                tar.add(backup_path, arcname=backup_path.name)
        
        elif compression == 'zip':
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(backup_path):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(backup_path.parent)
                        zipf.write(file_path, arcname)
        
        else:
            raise ValueError(f"Unsupported compression: {compression}")
        
        logger.info(f"Compression completed: {output_path}")
        return output_path
    
    async def _encrypt_backup(self, backup_path: Path) -> Path:
        """Encrypt backup file"""
        if not self.cipher:
            raise ValueError("Encryption not initialized")
        
        logger.info("Encrypting backup...")
        
        encrypted_path = backup_path.with_suffix(backup_path.suffix + '.enc')
        
        # Read file content
        with open(backup_path, 'rb') as f:
            data = f.read()
        
        # Encrypt data
        encrypted_data = self.cipher.encrypt(data)
        
        # Write encrypted data
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        logger.info(f"Encryption completed: {encrypted_path}")
        return encrypted_path
    
    async def _upload_to_cloud(self, backup_path: Path,
                              manifest_path: Path,
                              backup_id: str) -> Dict:
        """Upload backup to cloud storage (S3)"""
        if not self.cloud_config['s3_bucket']:
            logger.warning("Cloud backup configured but no S3 bucket specified")
            return {'status': 'skipped', 'reason': 'No S3 bucket'}
        
        if not self.cloud_config['aws_access_key']:
            logger.warning("AWS credentials not configured for cloud backup")
            return {'status': 'skipped', 'reason': 'No AWS credentials'}
        
        try:
            logger.info(f"Uploading backup to S3: {self.cloud_config['s3_bucket']}")
            
            # Initialize S3 session
            session = aioboto3.Session(
                aws_access_key_id=self.cloud_config['aws_access_key'],
                aws_secret_access_key=self.cloud_config['aws_secret_key'],
                region_name=self.cloud_config['aws_region']
            )
            
            async with session.client('s3') as s3:
                # Upload backup file
                backup_key = f"{self.cloud_config['s3_prefix']}{backup_path.name}"
                with open(backup_path, 'rb') as f:
                    await s3.upload_fileobj(f, self.cloud_config['s3_bucket'], backup_key)
                
                # Upload manifest
                manifest_key = f"{self.cloud_config['s3_prefix']}{manifest_path.name}"
                with open(manifest_path, 'rb') as f:
                    await s3.upload_fileobj(f, self.cloud_config['s3_bucket'], manifest_key)
                
                # Generate pre-signed URL (valid for 7 days)
                backup_url = await s3.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': self.cloud_config['s3_bucket'],
                        'Key': backup_key
                    },
                    ExpiresIn=604800  # 7 days
                )
                
                logger.info(f"Cloud upload completed: {backup_key}")
                
                return {
                    'status': 'success',
                    'bucket': self.cloud_config['s3_bucket'],
                    'backup_key': backup_key,
                    'manifest_key': manifest_key,
                    'presigned_url': backup_url,
                    'region': self.cloud_config['aws_region']
                }
                
        except Exception as e:
            logger.error(f"Cloud upload failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _get_last_backup(self, backup_type: str = None) -> Optional[Dict]:
        """Get information about the last backup"""
        backup_files = list(self.backup_dir.glob('*_manifest.json'))
        
        if not backup_files:
            return None
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        
        for manifest_file in backup_files:
            try:
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                
                if backup_type and manifest.get('backup_type') != backup_type:
                    continue
                
                return {
                    'backup_id': manifest.get('backup_id'),
                    'backup_type': manifest.get('backup_type'),
                    'timestamp': manifest.get('timestamp'),
                    'manifest_path': str(manifest_file),
                    'backup_file': self._find_backup_file(manifest.get('backup_id'))
                }
            except Exception:
                continue
        
        return None
    
    def _find_backup_file(self, backup_id: str) -> Optional[str]:
        """Find the actual backup file for a given backup ID"""
        patterns = [
            f"{backup_id}.tar.gz",
            f"{backup_id}.tar.bz2",
            f"{backup_id}.tar.xz",
            f"{backup_id}.zip",
            f"{backup_id}.tar.gz.enc",
            f"{backup_id}.tar.bz2.enc",
            f"{backup_id}.tar.xz.enc",
            f"{backup_id}.zip.enc",
            backup_id  # Directory backup
        ]
        
        for pattern in patterns:
            backup_path = self.backup_dir / pattern
            if backup_path.exists():
                return str(backup_path)
        
        return None
    
    def _update_stats(self, backup_details: Dict, success: bool):
        """Update backup statistics"""
        self.stats['total_backups'] += 1
        
        if success:
            self.stats['successful_backups'] += 1
            self.stats['last_backup'] = datetime.now().isoformat()
            
            if 'total_size' in backup_details:
                self.stats['total_size_gb'] += backup_details['total_size'] / (1024**3)
        else:
            self.stats['failed_backups'] += 1
    
    async def restore_backup(self, backup_id: str = None,
                           target_dir: Path = None,
                           restore_type: str = 'full') -> Dict:
        """
        Restore from backup
        
        Args:
            backup_id: ID of backup to restore (None for latest)
            target_dir: Target directory for restoration
            restore_type: Type of restoration (full, partial, data_only)
            
        Returns:
            Dictionary with restoration results
        """
        logger.info(f"Starting restoration: backup_id={backup_id}, type={restore_type}")
        
        # Determine backup to restore
        if backup_id is None:
            last_backup = self._get_last_backup()
            if not last_backup:
                return {'status': 'failed', 'error': 'No backups found'}
            backup_id = last_backup['backup_id']
            backup_info = last_backup
        else:
            backup_info = self._get_backup_info(backup_id)
            if not backup_info:
                return {'status': 'failed', 'error': f'Backup not found: {backup_id}'}
        
        # Set target directory
        if target_dir is None:
            target_dir = self.project_root / 'restored'
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Find backup file
            backup_path = Path(backup_info['backup_file'])
            if not backup_path.exists():
                return {'status': 'failed', 'error': f'Backup file not found: {backup_path}'}
            
            # Decrypt if needed
            if backup_path.suffix == '.enc':
                logger.info("Decrypting backup...")
                backup_path = await self._decrypt_backup(backup_path)
            
            # Decompress if needed
            if backup_path.suffix in ['.gz', '.bz2', '.xz', '.zip']:
                logger.info("Decompressing backup...")
                extracted_dir = await self._decompress_backup(backup_path, target_dir)
            else:
                # Directory backup
                extracted_dir = backup_path
            
            # Load manifest
            manifest_path = Path(backup_info['manifest_path'])
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Perform restoration based on type
            if restore_type == 'full':
                result = await self._restore_full(extracted_dir, target_dir, manifest)
            elif restore_type == 'data_only':
                result = await self._restore_data_only(extracted_dir, target_dir, manifest)
            elif restore_type == 'partial':
                result = await self._restore_partial(extracted_dir, target_dir, manifest)
            else:
                raise ValueError(f"Unknown restore type: {restore_type}")
            
            # Cleanup temporary files
            if extracted_dir != backup_path and extracted_dir.is_dir():
                shutil.rmtree(extracted_dir)
            
            if backup_path.suffix == '.enc':
                os.remove(backup_path)
            
            logger.info(f"Restoration completed successfully: {result['file_count']} files restored")
            
            return {
                'status': 'success',
                'backup_id': backup_id,
                'restore_type': restore_type,
                'target_dir': str(target_dir),
                'details': result
            }
            
        except Exception as e:
            logger.error(f"Restoration failed: {e}")
            return {
                'status': 'failed',
                'backup_id': backup_id,
                'error': str(e)
            }
    
    async def _decrypt_backup(self, encrypted_path: Path) -> Path:
        """Decrypt backup file"""
        if not self.cipher:
            raise ValueError("Encryption cipher not available")
        
        decrypted_path = encrypted_path.with_suffix('')  # Remove .enc suffix
        
        # Read encrypted data
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt data
        decrypted_data = self.cipher.decrypt(encrypted_data)
        
        # Write decrypted data
        with open(decrypted_path, 'wb') as f:
            f.write(decrypted_data)
        
        return decrypted_path
    
    async def _decompress_backup(self, compressed_path: Path,
                                target_dir: Path) -> Path:
        """Decompress backup file"""
        suffix = compressed_path.suffix
        
        if suffix == '.gz' or compressed_path.suffixes[-2:] == ['.tar', '.gz']:
            with tarfile.open(compressed_path, 'r:gz') as tar:
                tar.extractall(target_dir)
                # Get the extracted directory name
                members = tar.getmembers()
                if members:
                    return target_dir / members[0].name.split('/')[0]
        
        elif suffix == '.bz2' or compressed_path.suffixes[-2:] == ['.tar', '.bz2']:
            with tarfile.open(compressed_path, 'r:bz2') as tar:
                tar.extractall(target_dir)
                members = tar.getmembers()
                if members:
                    return target_dir / members[0].name.split('/')[0]
        
        elif suffix == '.xz' or compressed_path.suffixes[-2:] == ['.tar', '.xz']:
            with tarfile.open(compressed_path, 'r:xz') as tar:
                tar.extractall(target_dir)
                members = tar.getmembers()
                if members:
                    return target_dir / members[0].name.split('/')[0]
        
        elif suffix == '.zip':
            with zipfile.ZipFile(compressed_path, 'r') as zipf:
                zipf.extractall(target_dir)
                # Get the first directory in the zip
                namelist = zipf.namelist()
                if namelist:
                    first_item = namelist[0]
                    if '/' in first_item:
                        return target_dir / first_item.split('/')[0]
        
        return target_dir
    
    async def _restore_full(self, backup_dir: Path,
                           target_dir: Path,
                           manifest: Dict) -> Dict:
        """Perform full restoration"""
        result = {
            'type': 'full',
            'file_count': 0,
            'total_size': 0,
            'restored_directories': []
        }
        
        # Restore data directory
        data_backup = backup_dir / 'data'
        if data_backup.exists():
            target_data = target_dir / 'storage' / 'data'
            target_data.mkdir(parents=True, exist_ok=True)
            
            data_result = await self._restore_directory(data_backup, target_data)
            result['file_count'] += data_result['file_count']
            result['total_size'] += data_result['total_size']
            result['restored_directories'].append('data')
        
        # Restore logs directory
        logs_backup = backup_dir / 'logs'
        if logs_backup.exists():
            target_logs = target_dir / 'storage' / 'logs'
            target_logs.mkdir(parents=True, exist_ok=True)
            
            logs_result = await self._restore_directory(logs_backup, target_logs)
            result['file_count'] += logs_result['file_count']
            result['total_size'] += logs_result['total_size']
            result['restored_directories'].append('logs')
        
        # Restore reports directory
        reports_backup = backup_dir / 'reports'
        if reports_backup.exists():
            target_reports = target_dir / 'reports'
            target_reports.mkdir(parents=True, exist_ok=True)
            
            reports_result = await self._restore_directory(reports_backup, target_reports)
            result['file_count'] += reports_result['file_count']
            result['total_size'] += reports_result['total_size']
            result['restored_directories'].append('reports')
        
        # Restore configuration
        config_backup = backup_dir / 'config'
        if config_backup.exists():
            target_config = target_dir / 'config'
            target_config.mkdir(parents=True, exist_ok=True)
            
            config_result = await self._restore_directory(config_backup, target_config)
            result['file_count'] += config_result['file_count']
            result['total_size'] += config_result['total_size']
            result['restored_directories'].append('config')
        
        # Restore source code
        src_backup = backup_dir / 'src'
        if src_backup.exists():
            target_src = target_dir / 'src'
            target_src.mkdir(parents=True, exist_ok=True)
            
            src_result = await self._restore_directory(src_backup, target_src)
            result['file_count'] += src_result['file_count']
            result['total_size'] += src_result['total_size']
            result['restored_directories'].append('src')
        
        # Restore database dump if exists
        db_dump_path = backup_dir / 'database_dump.json'
        if db_dump_path.exists():
            with open(db_dump_path, 'r') as f:
                db_dump = json.load(f)
            
            # Restore predictions if available
            if 'predictions' in db_dump and 'data' in db_dump['predictions']:
                predictions_path = target_dir / 'storage' / 'data' / 'predictions.json'
                predictions_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(predictions_path, 'w') as f:
                    json.dump(db_dump['predictions']['data'], f, indent=2)
                
                result['file_count'] += 1
                result['total_size'] += os.path.getsize(predictions_path)
                result['database_restored'] = True
        
        return result
    
    async def _restore_data_only(self, backup_dir: Path,
                                target_dir: Path,
                                manifest: Dict) -> Dict:
        """Restore only data (predictions, historical data)"""
        result = {
            'type': 'data_only',
            'file_count': 0,
            'total_size': 0,
            'restored_directories': []
        }
        
        # Restore data directory
        data_backup = backup_dir / 'data'
        if data_backup.exists():
            target_data = target_dir / 'storage' / 'data'
            target_data.mkdir(parents=True, exist_ok=True)
            
            data_result = await self._restore_directory(data_backup, target_data)
            result['file_count'] += data_result['file_count']
            result['total_size'] += data_result['total_size']
            result['restored_directories'].append('data')
        
        # Restore database dump
        db_dump_path = backup_dir / 'database_dump.json'
        if db_dump_path.exists():
            with open(db_dump_path, 'r') as f:
                db_dump = json.load(f)
            
            if 'predictions' in db_dump and 'data' in db_dump['predictions']:
                predictions_path = target_dir / 'storage' / 'data' / 'predictions.json'
                predictions_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(predictions_path, 'w') as f:
                    json.dump(db_dump['predictions']['data'], f, indent=2)
                
                result['file_count'] += 1
                result['total_size'] += os.path.getsize(predictions_path)
                result['database_restored'] = True
        
        return result
    
    async def _restore_partial(self, backup_dir: Path,
                              target_dir: Path,
                              manifest: Dict) -> Dict:
        """Restore specific files or directories"""
        # This would be implemented based on specific requirements
        # For now, restore data only
        return await self._restore_data_only(backup_dir, target_dir, manifest)
    
    async def _restore_directory(self, source_dir: Path,
                                target_dir: Path) -> Dict:
        """Restore directory from backup"""
        result = {
            'file_count': 0,
            'total_size': 0
        }
        
        # Walk through backup directory
        for root, dirs, files in os.walk(source_dir):
            # Calculate relative path
            rel_path = Path(root).relative_to(source_dir)
            current_target = target_dir / rel_path
            
            # Create target directory
            current_target.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for file in files:
                source_file = Path(root) / file
                target_file = current_target / file
                
                # Copy file
                shutil.copy2(source_file, target_file)
                
                # Update statistics
                result['file_count'] += 1
                result['total_size'] += os.path.getsize(source_file)
        
        return result
    
    def _get_backup_info(self, backup_id: str) -> Optional[Dict]:
        """Get information about a specific backup"""
        manifest_pattern = f"*{backup_id}*manifest.json"
        manifest_files = list(self.backup_dir.glob(manifest_pattern))
        
        if not manifest_files:
            return None
        
        try:
            with open(manifest_files[0], 'r') as f:
                manifest = json.load(f)
            
            backup_file = self._find_backup_file(backup_id)
            if not backup_file:
                return None
            
            return {
                'backup_id': backup_id,
                'backup_type': manifest.get('backup_type'),
                'timestamp': manifest.get('timestamp'),
                'manifest_path': str(manifest_files[0]),
                'backup_file': backup_file,
                'description': manifest.get('description', ''),
                'file_count': len(manifest.get('file_list', [])),
                'total_size': manifest.get('backup_size', 0)
            }
        except Exception:
            return None
    
    async def list_backups(self, detailed: bool = False) -> List[Dict]:
        """List all available backups"""
        backups = []
        manifest_files = list(self.backup_dir.glob('*_manifest.json'))
        
        for manifest_file in manifest_files:
            try:
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                
                backup_id = manifest.get('backup_id')
                backup_file = self._find_backup_file(backup_id)
                
                if not backup_file:
                    continue
                
                backup_info = {
                    'backup_id': backup_id,
                    'backup_type': manifest.get('backup_type'),
                    'timestamp': manifest.get('timestamp'),
                    'description': manifest.get('description', ''),
                    'file_count': len(manifest.get('file_list', [])),
                    'total_size': manifest.get('backup_size', 0),
                    'compressed': 'compressed' in manifest.get('backup_format', ''),
                    'encrypted': manifest.get('encrypted', False),
                    'manifest_path': str(manifest_file),
                    'backup_path': backup_file,
                    'age_days': self._calculate_age_days(manifest.get('timestamp'))
                }
                
                if detailed:
                    backup_info['details'] = {
                        'system_info': manifest.get('system_info'),
                        'config_hash': manifest.get('config_hash'),
                        'file_list_sample': manifest.get('file_list', [])[:5]
                    }
                
                backups.append(backup_info)
                
            except Exception as e:
                logger.warning(f"Failed to read manifest {manifest_file}: {e}")
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return backups
    
    def _calculate_age_days(self, timestamp_str: str) -> float:
        """Calculate age of backup in days"""
        try:
            backup_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            age = datetime.now() - backup_time
            return age.days + age.seconds / 86400
        except Exception:
            return 0
    
    async def cleanup_old_backups(self) -> Dict:
        """Clean up old backups based on retention policy"""
        logger.info("Cleaning up old backups...")
        
        backups = await self.list_backups()
        retention_days = self.backup_config['retention_days']
        max_backups = self.backup_config['max_backups']
        
        backups_to_keep = []
        backups_to_delete = []
        
        # Separate backups by type
        full_backups = []
        other_backups = []
        
        for backup in backups:
            if backup['backup_type'] == 'full':
                full_backups.append(backup)
            else:
                other_backups.append(backup)
        
        # Keep all full backups within retention period
        for backup in full_backups:
            if backup['age_days'] <= retention_days:
                backups_to_keep.append(backup)
            else:
                backups_to_delete.append(backup)
        
        # For other backups, keep if they have a corresponding full backup
        # and are within retention period
        for backup in other_backups:
            # Find corresponding full backup
            has_full_backup = any(
                b['backup_type'] == 'full' and 
                b['timestamp'] < backup['timestamp']
                for b in backups_to_keep
            )
            
            if has_full_backup and backup['age_days'] <= retention_days:
                backups_to_keep.append(backup)
            else:
                backups_to_delete.append(backup)
        
        # Enforce max_backups limit
        if len(backups_to_keep) > max_backups:
            # Sort by age (oldest first) and remove excess
            backups_to_keep.sort(key=lambda x: x['age_days'], reverse=True)
            excess = backups_to_keep[max_backups:]
            backups_to_delete.extend(excess)
            backups_to_keep = backups_to_keep[:max_backups]
        
        # Delete old backups
        deleted_files = []
        deleted_size = 0
        
        for backup in backups_to_delete:
            try:
                # Delete backup file
                backup_path = Path(backup['backup_path'])
                if backup_path.exists():
                    deleted_size += os.path.getsize(backup_path)
                    os.remove(backup_path)
                    deleted_files.append(str(backup_path))
                
                # Delete manifest
                manifest_path = Path(backup['manifest_path'])
                if manifest_path.exists():
                    os.remove(manifest_path)
                    deleted_files.append(str(manifest_path))
                
                logger.info(f"Deleted backup: {backup['backup_id']} ({backup['age_days']:.1f} days old)")
                
            except Exception as e:
                logger.error(f"Failed to delete backup {backup['backup_id']}: {e}")
        
        logger.info(f"Cleanup completed: {len(deleted_files)//2} backups deleted, "
                   f"{self._format_size(deleted_size)} freed")
        
        return {
            'backups_kept': len(backups_to_keep),
            'backups_deleted': len(backups_to_delete),
            'deleted_files': deleted_files,
            'freed_space': deleted_size
        }
    
    async def verify_backup(self, backup_id: str) -> Dict:
        """Verify backup integrity"""
        backup_info = self._get_backup_info(backup_id)
        if not backup_info:
            return {'status': 'failed', 'error': f'Backup not found: {backup_id}'}
        
        try:
            backup_path = Path(backup_info['backup_path'])
            manifest_path = Path(backup_info['manifest_path'])
            
            # Check if files exist
            if not backup_path.exists():
                return {'status': 'failed', 'error': 'Backup file missing'}
            
            if not manifest_path.exists():
                return {'status': 'failed', 'error': 'Manifest file missing'}
            
            # Load manifest
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Verify file size
            actual_size = os.path.getsize(backup_path)
            expected_size = manifest.get('backup_size', 0)
            
            size_match = actual_size == expected_size
            if not size_match:
                logger.warning(f"Size mismatch: expected {expected_size}, got {actual_size}")
            
            # For compressed backups, verify they can be decompressed
            can_decompress = False
            temp_dir = None
            
            try:
                if backup_path.suffix in ['.gz', '.bz2', '.xz', '.zip']:
                    temp_dir = Path(tempfile.mkdtemp())
                    extracted = await self._decompress_backup(backup_path, temp_dir)
                    can_decompress = extracted.exists() and any(extracted.iterdir())
            finally:
                if temp_dir and temp_dir.exists():
                    shutil.rmtree(temp_dir)
            
            # Calculate checksum if not encrypted
            checksum_match = True
            if backup_path.suffix != '.enc':
                actual_checksum = self._calculate_checksum(backup_path)
                # Note: We don't have expected checksum in manifest
                # This would need to be stored during backup creation
            
            status = 'healthy' if size_match and can_decompress else 'corrupted'
            
            return {
                'status': 'success',
                'backup_id': backup_id,
                'verification': {
                    'status': status,
                    'size_match': size_match,
                    'can_decompress': can_decompress,
                    'actual_size': actual_size,
                    'expected_size': expected_size,
                    'file_exists': True,
                    'manifest_exists': True
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'backup_id': backup_id,
                'error': str(e)
            }
    
    async def get_backup_stats(self) -> Dict:
        """Get backup statistics"""
        backups = await self.list_backups()
        
        total_size = sum(b['total_size'] for b in backups)
        by_type = {}
        
        for backup in backups:
            backup_type = backup['backup_type']
            if backup_type not in by_type:
                by_type[backup_type] = {
                    'count': 0,
                    'total_size': 0,
                    'oldest': None,
                    'newest': None
                }
            
            by_type[backup_type]['count'] += 1
            by_type[backup_type]['total_size'] += backup['total_size']
            
            # Update oldest/newest
            backup_time = datetime.fromisoformat(backup['timestamp'].replace('Z', '+00:00'))
            if (by_type[backup_type]['oldest'] is None or 
                backup_time < by_type[backup_type]['oldest']):
                by_type[backup_type]['oldest'] = backup_time
            
            if (by_type[backup_type]['newest'] is None or 
                backup_time > by_type[backup_type]['newest']):
                by_type[backup_type]['newest'] = backup_time
        
        # Calculate disk usage
        backup_dir_size = sum(
            f.stat().st_size for f in self.backup_dir.rglob('*') if f.is_file()
        )
        
        return {
            'total_backups': len(backups),
            'total_size': total_size,
            'total_size_formatted': self._format_size(total_size),
            'backups_by_type': by_type,
            'disk_usage': {
                'backup_dir': self._format_size(backup_dir_size),
                'data_dir': self._format_size(self._get_dir_size(self.data_dir)),
                'project_dir': self._format_size(self._get_dir_size(self.project_root))
            },
            'retention_days': self.backup_config['retention_days'],
            'max_backups': self.backup_config['max_backups'],
            'last_backup': self.stats['last_backup'],
            'success_rate': (
                self.stats['successful_backups'] / self.stats['total_backups'] * 100
                if self.stats['total_backups'] > 0 else 0
            )
        }
    
    def _get_dir_size(self, directory: Path) -> int:
        """Calculate directory size"""
        if not directory.exists():
            return 0
        
        total_size = 0
        for file in directory.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
        
        return total_size
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format"""
        if size_bytes == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        i = 0
        
        while size_bytes >= 1024 and i < len(units) - 1:
            size_bytes /= 1024
            i += 1
        
        return f"{size_bytes:.2f} {units[i]}"
    
    async def schedule_backup(self) -> Dict:
        """Schedule automatic backup based on configuration"""
        schedule = self.backup_config['schedule']
        
        # Check if backup is needed based on schedule
        last_backup = self._get_last_backup()
        if last_backup:
            last_time = datetime.fromisoformat(last_backup['timestamp'].replace('Z', '+00:00'))
            time_since_last = datetime.now() - last_time
            
            if schedule == 'daily' and time_since_last.days < 1:
                return {'status': 'skipped', 'reason': 'Backup already done today'}
            elif schedule == 'weekly' and time_since_last.days < 7:
                return {'status': 'skipped', 'reason': 'Backup already done this week'}
            elif schedule == 'monthly' and time_since_last.days < 30:
                return {'status': 'skipped', 'reason': 'Backup already done this month'}
        
        # Determine backup type based on schedule
        if schedule == 'daily':
            backup_type = 'incremental'
        elif schedule == 'weekly':
            backup_type = 'differential'
        elif schedule == 'monthly':
            backup_type = 'full'
        else:
            backup_type = 'incremental'
        
        # Create backup
        description = f"Scheduled {schedule} backup"
        result = await self.create_backup(backup_type, description)
        
        # Cleanup old backups
        await self.cleanup_old_backups()
        
        return result


# Command line interface
async def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Over/Under Predictor Backup Manager')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create backup command
    create_parser = subparsers.add_parser('create', help='Create new backup')
    create_parser.add_argument('--type', '-t', default='incremental',
                              choices=['full', 'incremental', 'differential'],
                              help='Type of backup')
    create_parser.add_argument('--description', '-d', default='',
                              help='Description for the backup')
    
    # Restore backup command
    restore_parser = subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('--backup-id', '-b', help='Backup ID to restore')
    restore_parser.add_argument('--target', '-t', help='Target directory for restoration')
    restore_parser.add_argument('--type', default='full',
                               choices=['full', 'data_only', 'partial'],
                               help='Type of restoration')
    
    # List backups command
    list_parser = subparsers.add_parser('list', help='List available backups')
    list_parser.add_argument('--detailed', '-d', action='store_true',
                            help='Show detailed information')
    
    # Verify backup command
    verify_parser = subparsers.add_parser('verify', help='Verify backup integrity')
    verify_parser.add_argument('--backup-id', '-b', required=True,
                              help='Backup ID to verify')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old backups')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show backup statistics')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Run scheduled backup')
    
    args = parser.parse_args()
    
    # Initialize backup manager
    backup_manager = BackupManager(args.config)
    
    if args.command == 'create':
        result = await backup_manager.create_backup(args.type, args.description)
        print(json.dumps(result, indent=2))
    
    elif args.command == 'restore':
        result = await backup_manager.restore_backup(
            args.backup_id,
            Path(args.target) if args.target else None,
            args.type
        )
        print(json.dumps(result, indent=2))
    
    elif args.command == 'list':
        backups = await backup_manager.list_backups(args.detailed)
        if not backups:
            print("No backups found")
        else:
            for i, backup in enumerate(backups, 1):
                print(f"\n{i}. {backup['backup_id']}")
                print(f"   Type: {backup['backup_type']}")
                print(f"   Date: {backup['timestamp']}")
                print(f"   Size: {backup_manager._format_size(backup['total_size'])}")
                print(f"   Age: {backup['age_days']:.1f} days")
                print(f"   Description: {backup['description']}")
                
                if args.detailed:
                    print(f"   Path: {backup['backup_path']}")
                    print(f"   Compressed: {backup['compressed']}")
                    print(f"   Encrypted: {backup['encrypted']}")
    
    elif args.command == 'verify':
        result = await backup_manager.verify_backup(args.backup_id)
        print(json.dumps(result, indent=2))
    
    elif args.command == 'cleanup':
        result = await backup_manager.cleanup_old_backups()
        print(f"Cleanup completed:")
        print(f"  Backups kept: {result['backups_kept']}")
        print(f"  Backups deleted: {result['backups_deleted']}")
        print(f"  Freed space: {backup_manager._format_size(result['freed_space'])}")
    
    elif args.command == 'stats':
        stats = await backup_manager.get_backup_stats()
        print(f"\n📊 Backup Statistics")
        print(f"=====================")
        print(f"Total Backups: {stats['total_backups']}")
        print(f"Total Size: {stats['total_size_formatted']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Last Backup: {stats['last_backup'] or 'Never'}")
        print(f"\nBy Type:")
        for backup_type, type_stats in stats['backups_by_type'].items():
            print(f"  {backup_type}: {type_stats['count']} backups, "
                  f"{backup_manager._format_size(type_stats['total_size'])}")
        print(f"\nDisk Usage:")
        for dir_name, size in stats['disk_usage'].items():
            print(f"  {dir_name}: {size}")
        print(f"\nRetention: {stats['retention_days']} days")
        print(f"Max Backups: {stats['max_backups']}")
    
    elif args.command == 'schedule':
        result = await backup_manager.schedule_backup()
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()


# Quick test function
async def quick_test():
    """Quick test of the backup system"""
    print("Running quick backup test...")
    
    backup_manager = BackupManager()
    
    # Create a small backup
    print("\n1. Creating test backup...")
    result = await backup_manager.create_backup('incremental', 'Test backup')
    print(f"   Status: {result['status']}")
    if result['status'] == 'success':
        print(f"   Backup ID: {result['backup_id']}")
        print(f"   Size: {backup_manager._format_size(result['details'].get('total_size', 0))}")
    
    # List backups
    print("\n2. Listing backups...")
    backups = await backup_manager.list_backups()
    print(f"   Found {len(backups)} backups")
    
    # Get stats
    print("\n3. Getting statistics...")
    stats = await backup_manager.get_backup_stats()
    print(f"   Total backups: {stats['total_backups']}")
    print(f"   Total size: {stats['total_size_formatted']}")
    
    print("\n✅ Quick test completed!")


if __name__ == "__main__":
    # Run main function
    asyncio.run(main())
