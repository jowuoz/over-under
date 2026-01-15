"""
telegram_client.py - Telegram client for Over/Under Predictor system
Handles all Telegram Bot API communication for sending predictions and alerts
"""
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import json
import time
import os
from urllib.parse import urlencode

# Import models and formatters - USE ABSOLUTE IMPORTS
from src.predictor.models import Prediction, BatchPrediction, Game, ProbabilityMetrics, RiskAssessment
from src.predictor.formatter import PredictionFormatter, OutputFormat, AlertLevel
from src.notifier.telegram_client import TelegramMessage

@dataclass
class TelegramMessage:
    """Telegram message data structure"""
    chat_id: str
    text: str
    parse_mode: str = "HTML"
    disable_web_page_preview: bool = True
    disable_notification: bool = False
    reply_to_message_id: Optional[int] = None
    message_id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request"""
        data = {
            "chat_id": self.chat_id,
            "text": self.text,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": self.disable_web_page_preview,
            "disable_notification": self.disable_notification,
        }
        
        if self.reply_to_message_id:
            data["reply_to_message_id"] = self.reply_to_message_id
        
        return data


@dataclass
class TelegramUser:
    """Telegram user information"""
    id: int
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None
    language_code: Optional[str] = None
    is_bot: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TelegramUser':
        """Create from API response"""
        return cls(
            id=data.get('id'),
            first_name=data.get('first_name', ''),
            last_name=data.get('last_name'),
            username=data.get('username'),
            language_code=data.get('language_code'),
            is_bot=data.get('is_bot', False)
        )


@dataclass
class TelegramChat:
    """Telegram chat information"""
    id: int
    type: str  # 'private', 'group', 'supergroup', 'channel'
    title: Optional[str] = None
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TelegramChat':
        """Create from API response"""
        return cls(
            id=data.get('id'),
            type=data.get('type', 'private'),
            title=data.get('title'),
            username=data.get('username'),
            first_name=data.get('first_name'),
            last_name=data.get('last_name')
        )


@dataclass
class TelegramUpdate:
    """Telegram update/event"""
    update_id: int
    message: Optional[Dict] = None
    callback_query: Optional[Dict] = None
    inline_query: Optional[Dict] = None
    timestamp: datetime = field(default_factory=datetime.now)


class TelegramError(Exception):
    """Custom Telegram API error"""
    def __init__(self, message: str, error_code: Optional[int] = None, response: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.response = response
        super().__init__(self.message)


class TelegramClient:
    """
    Main Telegram Bot API client
    Handles all communication with Telegram Bot API
    """
    
    BASE_URL = "https://api.telegram.org"
    
    def __init__(self, bot_token: str, config: Optional[Dict] = None):
        """
        Initialize Telegram client
        
        Args:
            bot_token: Telegram Bot API token
            config: Client configuration
        """
        if not bot_token:
            raise ValueError("Bot token is required")
        
        self.bot_token = bot_token
        self.config = config or self._default_config()
        self.logger = logging.getLogger("notifier.telegram")
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_connected = False
        
        # Rate limiting
        self.rate_limits = {
            'messages_per_second': 30,
            'messages_per_minute': 20,
            'last_messages': [],
        }
        
        # Message tracking
        self.sent_messages = []
        self.failed_messages = []
        
        # Formatter
        self.formatter = PredictionFormatter()
        
        # Bot information (populated on connect)
        self.bot_info: Optional[Dict] = None
        self.bot_username: Optional[str] = None
        
        # Chat management
        self.subscribed_chats: List[str] = []
        
        # Command handlers
        self.command_handlers = {
            'start': self._handle_start_command,
            'help': self._handle_help_command,
            'status': self._handle_status_command,
            'predictions': self._handle_predictions_command,
            'subscribe': self._handle_subscribe_command,
            'unsubscribe': self._handle_unsubscribe_command,
            'alerts': self._handle_alerts_command,
            'settings': self._handle_settings_command,
        }
        
        self.logger.info(f"Telegram client initialized with token: {bot_token[:10]}...")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'timeout': 10,
            'max_retries': 3,
            'retry_delay': 1,
            'max_message_length': 4096,
            'parse_mode': 'HTML',
            'disable_notifications': False,
            'enable_logging': True,
            'alert_thresholds': {
                'high': 0.75,
                'medium': 0.65,
                'low': 0.50,
            },
            'commands': {
                'start': 'Start the bot and get welcome message',
                'help': 'Show available commands',
                'status': 'Check bot status and system health',
                'predictions': 'Get latest predictions',
                'subscribe': 'Subscribe to alerts',
                'unsubscribe': 'Unsubscribe from alerts',
                'alerts': 'Configure alert settings',
                'settings': 'Change bot settings',
            }
        }
    
    async def connect(self) -> bool:
        """
        Connect to Telegram API and get bot information
        
        Returns:
            True if connected successfully
        """
        try:
            self.logger.info("Connecting to Telegram API...")
            
            # Create session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
            )
            
            # Get bot information
            self.bot_info = await self._make_request("getMe")
            
            if self.bot_info and self.bot_info.get('ok'):
                bot_data = self.bot_info['result']
                self.bot_username = bot_data.get('username')
                self.is_connected = True
                
                self.logger.info(f"Connected as @{self.bot_username} (ID: {bot_data.get('id')})")
                
                # Set bot commands
                await self._set_bot_commands()
                
                return True
            else:
                self.logger.error("Failed to get bot information")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Telegram API"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            self.is_connected = False
            self.logger.info("Disconnected from Telegram API")
    
    async def _make_request(
        self, 
        method: str, 
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make request to Telegram Bot API
        
        Args:
            method: API method name
            data: Request data
            files: Files to upload
            retry_count: Current retry attempt
            
        Returns:
            API response as dictionary
        """
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
            )
        
        url = f"{self.BASE_URL}/bot{self.bot_token}/{method}"
        
        # Rate limiting
        await self._respect_rate_limit()
        
        try:
            self.logger.debug(f"Making request to {method}")
            
            if files:
                # For file uploads
                form_data = aiohttp.FormData()
                
                # Add text data
                if data:
                    for key, value in data.items():
                        if value is not None:
                            form_data.add_field(key, str(value))
                
                # Add files
                for field_name, file_info in files.items():
                    if isinstance(file_info, tuple):
                        # (file_path, filename, content_type)
                        if len(file_info) == 3:
                            file_path, filename, content_type = file_info
                            form_data.add_field(
                                field_name,
                                open(file_path, 'rb'),
                                filename=filename,
                                content_type=content_type
                            )
                    elif isinstance(file_info, dict):
                        # {'path': 'file.txt', 'filename': 'file.txt', 'content_type': 'text/plain'}
                        form_data.add_field(
                            field_name,
                            open(file_info['path'], 'rb'),
                            filename=file_info.get('filename'),
                            content_type=file_info.get('content_type', 'application/octet-stream')
                        )
                
                async with self.session.post(url, data=form_data) as response:
                    response_text = await response.text()
                    result = json.loads(response_text) if response_text else {}
            else:
                # For regular requests
                async with self.session.post(url, json=data) as response:
                    response_text = await response.text()
                    result = json.loads(response_text) if response_text else {}
            
            # Check for errors
            if not result.get('ok'):
                error_msg = result.get('description', 'Unknown error')
                error_code = result.get('error_code')
                
                self.logger.error(f"Telegram API error ({method}): {error_msg} (code: {error_code})")
                
                # Handle rate limiting
                if error_code == 429:
                    retry_after = result.get('parameters', {}).get('retry_after', 30)
                    self.logger.warning(f"Rate limited, retrying after {retry_after} seconds")
                    await asyncio.sleep(retry_after)
                    
                    if retry_count < self.config['max_retries']:
                        return await self._make_request(method, data, files, retry_count + 1)
                
                # Handle other retryable errors
                elif error_code in [500, 502, 503, 504] and retry_count < self.config['max_retries']:
                    retry_delay = self.config['retry_delay'] * (2 ** retry_count)
                    self.logger.warning(f"Server error {error_code}, retrying in {retry_delay}s")
                    await asyncio.sleep(retry_delay)
                    return await self._make_request(method, data, files, retry_count + 1)
                
                raise TelegramError(f"Telegram API error: {error_msg}", error_code, result)
            
            return result
            
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error in {method}: {e}")
            
            if retry_count < self.config['max_retries']:
                retry_delay = self.config['retry_delay'] * (2 ** retry_count)
                self.logger.info(f"Retrying {method} in {retry_delay}s (attempt {retry_count + 1})")
                await asyncio.sleep(retry_delay)
                return await self._make_request(method, data, files, retry_count + 1)
            
            raise TelegramError(f"Network error: {e}")
        
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {method}: {e}")
            raise TelegramError(f"Invalid response from Telegram API: {e}")
        
        except Exception as e:
            self.logger.error(f"Unexpected error in {method}: {e}")
            raise TelegramError(f"Unexpected error: {e}")
    
    async def _respect_rate_limit(self):
        """Respect Telegram rate limits"""
        now = time.time()
        
        # Clean old messages
        self.rate_limits['last_messages'] = [
            ts for ts in self.rate_limits['last_messages']
            if now - ts < 60  # Keep only last minute
        ]
        
        # Check per-second limit
        recent_second = [ts for ts in self.rate_limits['last_messages'] if now - ts < 1]
        if len(recent_second) >= self.rate_limits['messages_per_second']:
            wait_time = 1 - (now - recent_second[0])
            if wait_time > 0:
                self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s (per-second limit)")
                await asyncio.sleep(wait_time)
        
        # Check per-minute limit
        if len(self.rate_limits['last_messages']) >= self.rate_limits['messages_per_minute']:
            oldest = self.rate_limits['last_messages'][0]
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s (per-minute limit)")
                await asyncio.sleep(wait_time)
        
        # Add current message timestamp
        self.rate_limits['last_messages'].append(now)
    
    async def _set_bot_commands(self):
        """Set bot commands for menu"""
        commands = []
        
        for command, description in self.config['commands'].items():
            commands.append({
                "command": command,
                "description": description
            })
        
        try:
            await self._make_request("setMyCommands", {
                "commands": json.dumps(commands),
                "scope": {"type": "default"},
                "language_code": "en"
            })
            self.logger.debug("Bot commands set successfully")
        except Exception as e:
            self.logger.warning(f"Failed to set bot commands: {e}")
    
    # Core messaging methods
    
    async def send_message(self, message: TelegramMessage) -> Dict[str, Any]:
        """
        Send a message to Telegram chat
        
        Args:
            message: TelegramMessage object
            
        Returns:
            API response
        """
        try:
            # Ensure message length is within limits
            if len(message.text) > self.config['max_message_length']:
                self.logger.warning(f"Message too long ({len(message.text)} chars), truncating")
                message.text = message.text[:self.config['max_message_length'] - 100] + "\n...\n[Message truncated]"
            
            response = await self._make_request("sendMessage", message.to_dict())
            
            if response.get('ok'):
                message_data = response['result']
                message.message_id = message_data.get('message_id')
                
                # Track successful message
                self.sent_messages.append({
                    'message_id': message.message_id,
                    'chat_id': message.chat_id,
                    'timestamp': message.timestamp,
                    'text_length': len(message.text)
                })
                
                self.logger.debug(f"Message sent to chat {message.chat_id} (ID: {message.message_id})")
            else:
                self.failed_messages.append({
                    'chat_id': message.chat_id,
                    'timestamp': message.timestamp,
                    'error': response.get('description', 'Unknown error')
                })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to send message to {message.chat_id}: {e}")
            self.failed_messages.append({
                'chat_id': message.chat_id,
                'timestamp': message.timestamp,
                'error': str(e)
            })
            raise
    
    async def send_prediction(self, prediction: Prediction, chat_id: str, alert_level: Optional[AlertLevel] = None) -> bool:
        """
        Send a prediction as Telegram message
        
        Args:
            prediction: Prediction object
            chat_id: Target chat ID
            alert_level: Optional alert level override
            
        Returns:
            True if sent successfully
        """
        try:
            # Format prediction for Telegram
            formatted_text = self.formatter.format_prediction(
                prediction, 
                OutputFormat.TELEGRAM,
                alert_level
            )
            
            # Create message
            message = TelegramMessage(
                chat_id=chat_id,
                text=formatted_text,
                parse_mode="HTML",
                disable_web_page_preview=True,
                disable_notification=False
            )
            
            # Send message
            response = await self.send_message(message)
            
            return response.get('ok', False)
            
        except Exception as e:
            self.logger.error(f"Failed to send prediction to {chat_id}: {e}")
            return False
    
    async def send_batch_predictions(self, batch: BatchPrediction, chat_id: str, include_all: bool = False) -> bool:
        """
        Send batch of predictions
        
        Args:
            batch: BatchPrediction object
            chat_id: Target chat ID
            include_all: Send all predictions or just high priority
            
        Returns:
            True if sent successfully
        """
        try:
            # Format batch for Telegram
            formatted_text = self.formatter.format_batch(
                batch,
                OutputFormat.TELEGRAM,
                include_all
            )
            
            # Create message
            message = TelegramMessage(
                chat_id=chat_id,
                text=formatted_text,
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            
            # Send message
            response = await self.send_message(message)
            
            return response.get('ok', False)
            
        except Exception as e:
            self.logger.error(f"Failed to send batch predictions to {chat_id}: {e}")
            return False
    
    async def send_alert(self, prediction: Prediction, chat_id: str) -> bool:
        """
        Send alert for high-probability prediction
        
        Args:
            prediction: Prediction object
            chat_id: Target chat ID
            
        Returns:
            True if sent successfully
        """
        try:
            # Only send alerts for high priority predictions
            if prediction.alert_priority < 50:
                self.logger.debug(f"Skipping alert for low priority prediction: {prediction.id}")
                return False
            
            # Determine alert level
            alert_level = AlertLevel.HIGH if prediction.alert_priority >= 80 else AlertLevel.MEDIUM
            
            # Send as prediction with alert styling
            return await self.send_prediction(prediction, chat_id, alert_level)
            
        except Exception as e:
            self.logger.error(f"Failed to send alert to {chat_id}: {e}")
            return False
    
    async def send_bulk_alerts(self, predictions: List[Prediction], chat_id: str, max_alerts: int = 5) -> Dict[str, Any]:
        """
        Send multiple alerts in one go
        
        Args:
            predictions: List of Prediction objects
            chat_id: Target chat ID
            max_alerts: Maximum number of alerts to send
            
        Returns:
            Dictionary with results
        """
        results = {
            'total': len(predictions),
            'sent': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
        
        # Sort by alert priority
        sorted_predictions = sorted(predictions, key=lambda p: p.alert_priority, reverse=True)
        
        # Send top predictions
        for i, prediction in enumerate(sorted_predictions[:max_alerts]):
            try:
                success = await self.send_alert(prediction, chat_id)
                
                result_detail = {
                    'prediction_id': prediction.id,
                    'priority': prediction.alert_priority,
                    'success': success
                }
                
                if success:
                    results['sent'] += 1
                    self.logger.info(f"Alert {i+1}/{min(len(predictions), max_alerts)} sent successfully")
                else:
                    results['failed'] += 1
                    self.logger.warning(f"Alert {i+1}/{min(len(predictions), max_alerts)} failed")
                
                results['details'].append(result_detail)
                
                # Small delay between alerts to avoid rate limiting
                if i < len(sorted_predictions[:max_alerts]) - 1:
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                results['failed'] += 1
                self.logger.error(f"Error sending alert {i+1}: {e}")
                results['details'].append({
                    'prediction_id': prediction.id,
                    'priority': prediction.alert_priority,
                    'success': False,
                    'error': str(e)
                })
        
        results['skipped'] = max(0, len(predictions) - max_alerts)
        
        # Send summary if we sent any alerts
        if results['sent'] > 0:
            summary = (
                f"📊 <b>Alert Summary</b>\n\n"
                f"✅ Successfully sent: {results['sent']}\n"
                f"❌ Failed: {results['failed']}\n"
                f"⏭️ Skipped (low priority): {results['skipped']}\n\n"
                f"Total predictions analyzed: {results['total']}"
            )
            
            try:
                await self.send_message(TelegramMessage(
                    chat_id=chat_id,
                    text=summary,
                    parse_mode="HTML",
                    disable_notification=True
                ))
            except Exception as e:
                self.logger.warning(f"Failed to send alert summary: {e}")
        
        return results
    
    # Advanced messaging methods
    
    async def edit_message(self, chat_id: str, message_id: int, new_text: str) -> bool:
        """
        Edit an existing message
        
        Args:
            chat_id: Chat ID
            message_id: Message ID to edit
            new_text: New message text
            
        Returns:
            True if edited successfully
        """
        try:
            response = await self._make_request("editMessageText", {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": new_text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            })
            
            return response.get('ok', False)
            
        except Exception as e:
            self.logger.error(f"Failed to edit message {message_id} in chat {chat_id}: {e}")
            return False
    
    async def delete_message(self, chat_id: str, message_id: int) -> bool:
        """
        Delete a message
        
        Args:
            chat_id: Chat ID
            message_id: Message ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            response = await self._make_request("deleteMessage", {
                "chat_id": chat_id,
                "message_id": message_id
            })
            
            return response.get('ok', False)
            
        except Exception as e:
            self.logger.error(f"Failed to delete message {message_id} from chat {chat_id}: {e}")
            return False
    
    async def send_photo(self, chat_id: str, photo_path: str, caption: str = "") -> bool:
        """
        Send a photo
        
        Args:
            chat_id: Chat ID
            photo_path: Path to photo file
            caption: Photo caption
            
        Returns:
            True if sent successfully
        """
        try:
            if not os.path.exists(photo_path):
                self.logger.error(f"Photo file not found: {photo_path}")
                return False
            
            files = {
                "photo": (photo_path, "image/png")
            }
            
            response = await self._make_request("sendPhoto", {
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": "HTML"
            }, files=files)
            
            return response.get('ok', False)
            
        except Exception as e:
            self.logger.error(f"Failed to send photo to {chat_id}: {e}")
            return False
    
    async def send_document(self, chat_id: str, document_path: str, caption: str = "") -> bool:
        """
        Send a document
        
        Args:
            chat_id: Chat ID
            document_path: Path to document file
            caption: Document caption
            
        Returns:
            True if sent successfully
        """
        try:
            if not os.path.exists(document_path):
                self.logger.error(f"Document file not found: {document_path}")
                return False
            
            filename = os.path.basename(document_path)
            
            files = {
                "document": (document_path, "application/octet-stream")
            }
            
            response = await self._make_request("sendDocument", {
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": "HTML"
            }, files=files)
            
            return response.get('ok', False)
            
        except Exception as e:
            self.logger.error(f"Failed to send document to {chat_id}: {e}")
            return False
    
    # Chat and user management
    
    async def get_chat_info(self, chat_id: str) -> Optional[TelegramChat]:
        """
        Get information about a chat
        
        Args:
            chat_id: Chat ID
            
        Returns:
            TelegramChat object or None
        """
        try:
            response = await self._make_request("getChat", {
                "chat_id": chat_id
            })
            
            if response.get('ok'):
                return TelegramChat.from_dict(response['result'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get chat info for {chat_id}: {e}")
            return None
    
    async def get_chat_members_count(self, chat_id: str) -> int:
        """
        Get number of members in a chat
        
        Args:
            chat_id: Chat ID
            
        Returns:
            Number of members
        """
        try:
            response = await self._make_request("getChatMembersCount", {
                "chat_id": chat_id
            })
            
            if response.get('ok'):
                return response['result']
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to get chat members count for {chat_id}: {e}")
            return 0
    
    async def get_updates(self, offset: Optional[int] = None, timeout: int = 30) -> List[TelegramUpdate]:
        """
        Get new updates from Telegram
        
        Args:
            offset: Identifier of the first update to return
            timeout: Timeout in seconds
            
        Returns:
            List of TelegramUpdate objects
        """
        try:
            params = {
                "timeout": timeout
            }
            
            if offset:
                params["offset"] = offset
            
            response = await self._make_request("getUpdates", params)
            
            if response.get('ok'):
                updates = []
                for update_data in response['result']:
                    update = TelegramUpdate(
                        update_id=update_data.get('update_id'),
                        message=update_data.get('message'),
                        callback_query=update_data.get('callback_query'),
                        inline_query=update_data.get('inline_query')
                    )
                    updates.append(update)
                
                return updates
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get updates: {e}")
            return []
    
    # Command handlers
    
    async def _handle_start_command(self, chat_id: str, user: Optional[TelegramUser] = None):
        """Handle /start command"""
        welcome_message = (
            f"👋 <b>Welcome to Over/Under Predictor Bot!</b>\n\n"
            f"I'm @{self.bot_username}, your AI-powered football prediction assistant.\n\n"
            f"<b>What I can do:</b>\n"
            f"• ⚽ Analyze live football matches\n"
            f"• 📊 Calculate Over/Under probabilities\n"
            f"• 🚨 Send high-probability alerts\n"
            f"• 📈 Provide statistical insights\n\n"
            f"<b>Available Commands:</b>\n"
            f"/help - Show all commands\n"
            f"/predictions - Get latest predictions\n"
            f"/alerts - Configure alert settings\n"
            f"/subscribe - Subscribe to alerts\n"
            f"/status - Check system status\n\n"
            f"<i>I'll automatically send you alerts for high-probability opportunities!</i>"
        )
        
        if user and user.first_name:
            welcome_message = f"Hey {user.first_name}!\n\n" + welcome_message
        
        await self.send_message(TelegramMessage(
            chat_id=chat_id,
            text=welcome_message,
            parse_mode="HTML"
        ))
    
    async def _handle_help_command(self, chat_id: str):
        """Handle /help command"""
        help_message = (
            f"🤖 <b>Over/Under Predictor Bot - Help</b>\n\n"
            f"<b>Available Commands:</b>\n"
        )
        
        for command, description in self.config['commands'].items():
            help_message += f"/{command} - {description}\n"
        
        help_message += (
            f"\n<b>How it works:</b>\n"
            f"1. I analyze live football matches using multiple statistical models\n"
            f"2. I calculate probabilities for Over/Under goals markets\n"
            f"3. I send alerts when high-probability opportunities are detected\n"
            f"4. You can configure alert thresholds and frequency\n\n"
            f"<b>Alert Levels:</b>\n"
            f"🚨 High - >{self.config['alert_thresholds']['high']*100:.0f}% probability\n"
            f"📢 Medium - >{self.config['alert_thresholds']['medium']*100:.0f}% probability\n"
            f"📋 Low - >{self.config['alert_thresholds']['low']*100:.0f}% probability\n\n"
            f"<i>Use /subscribe to start receiving alerts!</i>"
        )
        
        await self.send_message(TelegramMessage(
            chat_id=chat_id,
            text=help_message,
            parse_mode="HTML"
        ))
    
    async def _handle_status_command(self, chat_id: str):
        """Handle /status command"""
        # Get statistics
        total_messages = len(self.sent_messages)
        failed_messages = len(self.failed_messages)
        success_rate = (total_messages - failed_messages) / total_messages if total_messages > 0 else 1.0
        
        # Get uptime
        if hasattr(self, '_start_time'):
            uptime = datetime.now() - self._start_time
            uptime_str = str(uptime).split('.')[0]
        else:
            uptime_str = "Unknown"
        
        status_message = (
            f"📊 <b>System Status</b>\n\n"
            f"<b>Bot Status:</b> {'🟢 Online' if self.is_connected else '🔴 Offline'}\n"
            f"<b>Bot Username:</b> @{self.bot_username}\n"
            f"<b>Uptime:</b> {uptime_str}\n\n"
            f"<b>Message Statistics:</b>\n"
            f"✅ Sent: {total_messages}\n"
            f"❌ Failed: {failed_messages}\n"
            f"📈 Success Rate: {success_rate:.1%}\n\n"
            f"<b>Rate Limits:</b>\n"
            f"Messages/sec: {len([m for m in self.rate_limits['last_messages'] if time.time() - m < 1])}\n"
            f"Messages/min: {len(self.rate_limits['last_messages'])}\n\n"
            f"<b>Subscribed Chats:</b> {len(self.subscribed_chats)}\n\n"
            f"<i>Last updated: {datetime.now().strftime('%H:%M:%S')}</i>"
        )
        
        await self.send_message(TelegramMessage(
            chat_id=chat_id,
            text=status_message,
            parse_mode="HTML"
        ))
    
    async def _handle_predictions_command(self, chat_id: str):
        """Handle /predictions command"""
        # This would typically fetch current predictions from your system
        # For now, send a placeholder message
        
        placeholder_message = (
            f"📈 <b>Latest Predictions</b>\n\n"
            f"Currently analyzing matches...\n\n"
            f"<i>Predictions are generated every 5 minutes based on live match data.</i>\n"
            f"<i>You'll receive automatic alerts for high-probability opportunities.</i>\n\n"
            f"<b>Next update in:</b> ~3 minutes\n\n"
            f"Use /alerts to configure your alert preferences."
        )
        
        await self.send_message(TelegramMessage(
            chat_id=chat_id,
            text=placeholder_message,
            parse_mode="HTML"
        ))
    
    async def _handle_subscribe_command(self, chat_id: str):
        """Handle /subscribe command"""
        if chat_id in self.subscribed_chats:
            await self.send_message(TelegramMessage(
                chat_id=chat_id,
                text="✅ <b>You're already subscribed!</b>\n\nYou'll receive alerts for high-probability predictions.",
                parse_mode="HTML"
            ))
        else:
            self.subscribed_chats.append(chat_id)
            await self.send_message(TelegramMessage(
                chat_id=chat_id,
                text="🎉 <b>Subscription Successful!</b>\n\nYou'll now receive alerts for high-probability Over/Under predictions.\n\nUse /alerts to configure your preferences.",
                parse_mode="HTML"
            ))
    
    async def _handle_unsubscribe_command(self, chat_id: str):
        """Handle /unsubscribe command"""
        if chat_id in self.subscribed_chats:
            self.subscribed_chats.remove(chat_id)
            await self.send_message(TelegramMessage(
                chat_id=chat_id,
                text="🔕 <b>Unsubscribed Successfully</b>\n\nYou'll no longer receive prediction alerts.\n\nUse /subscribe to re-enable alerts.",
                parse_mode="HTML"
            ))
        else:
            await self.send_message(TelegramMessage(
                chat_id=chat_id,
                text="ℹ️ <b>Not Subscribed</b>\n\nYou're not currently subscribed to alerts.\n\nUse /subscribe to start receiving predictions.",
                parse_mode="HTML"
            ))
    
    async def _handle_alerts_command(self, chat_id: str):
        """Handle /alerts command"""
        alert_settings = (
            f"🔔 <b>Alert Settings</b>\n\n"
            f"<b>Current Status:</b> {'Subscribed ✅' if chat_id in self.subscribed_chats else 'Not Subscribed ❌'}\n\n"
            f"<b>Alert Thresholds:</b>\n"
            f"🚨 High: >{self.config['alert_thresholds']['high']*100:.0f}% probability\n"
            f"📢 Medium: >{self.config['alert_thresholds']['medium']*100:.0f}% probability\n"
            f"📋 Low: >{self.config['alert_thresholds']['low']*100:.0f}% probability\n\n"
            f"<b>Commands:</b>\n"
            f"/subscribe - Enable alerts\n"
            f"/unsubscribe - Disable alerts\n\n"
            f"<i>Alerts are sent only for high-confidence predictions.</i>"
        )
        
        await self.send_message(TelegramMessage(
            chat_id=chat_id,
            text=alert_settings,
            parse_mode="HTML"
        ))
    
    async def _handle_settings_command(self, chat_id: str):
        """Handle /settings command"""
        settings_message = (
            f"⚙️ <b>Bot Settings</b>\n\n"
            f"<b>Current Configuration:</b>\n"
            f"• Parse Mode: {self.config['parse_mode']}\n"
            f"• Max Message Length: {self.config['max_message_length']} chars\n"
            f"• Timeout: {self.config['timeout']} seconds\n"
            f"• Max Retries: {self.config['max_retries']}\n\n"
            f"<b>Alert Thresholds:</b>\n"
            f"• High: {self.config['alert_thresholds']['high']*100:.0f}%\n"
            f"• Medium: {self.config['alert_thresholds']['medium']*100:.0f}%\n"
            f"• Low: {self.config['alert_thresholds']['low']*100:.0f}%\n\n"
            f"<i>Settings are configured in the bot configuration file.</i>"
        )
        
        await self.send_message(TelegramMessage(
            chat_id=chat_id,
            text=settings_message,
            parse_mode="HTML"
        ))
    
    async def handle_message(self, message_data: Dict):
        """
        Handle incoming message from Telegram
        
        Args:
            message_data: Telegram message data
        """
        try:
            chat_id = message_data.get('chat', {}).get('id')
            text = message_data.get('text', '').strip()
            user_data = message_data.get('from', {})
            
            if not chat_id or not text:
                return
            
            user = TelegramUser.from_dict(user_data) if user_data else None
            
            self.logger.info(f"Received message from {user.first_name if user else 'Unknown'}: {text}")
            
            # Check if it's a command
            if text.startswith('/'):
                command = text[1:].split(' ')[0].lower()  # Get command without parameters
                
                if command in self.command_handlers:
                    await self.command_handlers[command](chat_id, user)
                else:
                    await self.send_message(TelegramMessage(
                        chat_id=chat_id,
                        text=f"❓ <b>Unknown Command</b>\n\nCommand '{command}' not recognized.\n\nUse /help to see available commands.",
                        parse_mode="HTML"
                    ))
            else:
                # Handle regular message
                response = (
                    f"👋 Hi {user.first_name if user else 'there'}!\n\n"
                    f"I'm a prediction bot focused on Over/Under football markets.\n\n"
                    f"Use /help to see what I can do, or /predictions to get the latest predictions."
                )
                
                await self.send_message(TelegramMessage(
                    chat_id=chat_id,
                    text=response,
                    parse_mode="HTML"
                ))
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    # Batch operations and utilities
    
    async def send_to_all_subscribed(self, message: str) -> Dict[str, Any]:
        """
        Send message to all subscribed chats
        
        Args:
            message: Message text
            
        Returns:
            Dictionary with results
        """
        results = {
            'total': len(self.subscribed_chats),
            'sent': 0,
            'failed': 0,
            'details': []
        }
        
        for chat_id in self.subscribed_chats:
            try:
                telegram_message = TelegramMessage(
                    chat_id=chat_id,
                    text=message,
                    parse_mode="HTML"
                )
                
                response = await self.send_message(telegram_message)
                
                results['details'].append({
                    'chat_id': chat_id,
                    'success': response.get('ok', False)
                })
                
                if response.get('ok'):
                    results['sent'] += 1
                else:
                    results['failed'] += 1
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Failed to send to {chat_id}: {e}")
                results['failed'] += 1
                results['details'].append({
                    'chat_id': chat_id,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def broadcast_predictions(self, batch: BatchPrediction, only_high_priority: bool = True) -> Dict[str, Any]:
        """
        Broadcast predictions to all subscribed chats
        
        Args:
            batch: BatchPrediction object
            only_high_priority: Send only high priority predictions
            
        Returns:
            Dictionary with results
        """
        results = {
            'total_chats': len(self.subscribed_chats),
            'chats_reached': 0,
            'total_predictions': len(batch.predictions),
            'predictions_sent': 0,
            'details': []
        }
        
        if not self.subscribed_chats:
            self.logger.warning("No subscribed chats to broadcast to")
            return results
        
        # Format the batch message
        formatted_text = self.formatter.format_batch(
            batch,
            OutputFormat.TELEGRAM,
            not only_high_priority
        )
        
        # Send to all subscribed chats
        for chat_id in self.subscribed_chats:
            try:
                telegram_message = TelegramMessage(
                    chat_id=chat_id,
                    text=formatted_text,
                    parse_mode="HTML"
                )
                
                response = await self.send_message(telegram_message)
                
                if response.get('ok'):
                    results['chats_reached'] += 1
                    results['predictions_sent'] += len(
                        batch.get_high_priority_predictions() if only_high_priority 
                        else batch.predictions
                    )
                
                results['details'].append({
                    'chat_id': chat_id,
                    'success': response.get('ok', False),
                    'message': 'Broadcast successful' if response.get('ok') else response.get('description', 'Unknown error')
                })
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.2)
                
            except Exception as e:
                self.logger.error(f"Failed to broadcast to {chat_id}: {e}")
                results['details'].append({
                    'chat_id': chat_id,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get client statistics
        
        Returns:
            Dictionary with statistics
        """
        now = datetime.now()
        
        # Messages in last hour
        recent_messages = [
            msg for msg in self.sent_messages
            if (now - msg['timestamp']).seconds < 3600
        ]
        
        # Failed messages in last hour
        recent_failed = [
            msg for msg in self.failed_messages
            if (now - msg['timestamp']).seconds < 3600
        ]
        
        success_rate = len(recent_messages) / (len(recent_messages) + len(recent_failed)) if (len(recent_messages) + len(recent_failed)) > 0 else 1.0
        
        return {
            'bot_username': self.bot_username,
            'is_connected': self.is_connected,
            'total_messages_sent': len(self.sent_messages),
            'total_messages_failed': len(self.failed_messages),
            'recent_messages_hour': len(recent_messages),
            'recent_failed_hour': len(recent_failed),
            'success_rate_hour': success_rate,
            'subscribed_chats': len(self.subscribed_chats),
            'rate_limit_messages_minute': len(self.rate_limits['last_messages']),
            'config': {
                'timeout': self.config['timeout'],
                'max_retries': self.config['max_retries'],
                'max_message_length': self.config['max_message_length'],
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check
        
        Returns:
            Health status dictionary
        """
        try:
            # Test connection
            test_response = await self._make_request("getMe")
            
            health_status = {
                'status': 'healthy' if test_response.get('ok') else 'unhealthy',
                'bot_online': test_response.get('ok', False),
                'bot_username': self.bot_username,
                'session_active': self.session is not None and not self.session.closed,
                'subscribed_chats': len(self.subscribed_chats),
                'timestamp': datetime.now().isoformat()
            }
            
            if not test_response.get('ok'):
                health_status['error'] = test_response.get('description', 'Unknown error')
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Factory function and simplified interface

def create_telegram_client(bot_token: str, config: Optional[Dict] = None) -> TelegramClient:
    """
    Create Telegram client instance
    
    Args:
        bot_token: Telegram Bot API token
        config: Optional configuration
        
    Returns:
        TelegramClient instance
    """
    return TelegramClient(bot_token, config)


class TelegramNotifier:
    """
    Simplified interface for sending notifications via Telegram
    This is the main class you'll use in your prediction system
    """
    
    def __init__(self, bot_token: str, default_chat_id: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram Bot API token
            default_chat_id: Default chat ID for notifications
            config: Configuration dictionary
        """
        self.client = create_telegram_client(bot_token, config)
        self.default_chat_id = default_chat_id
        self.logger = logging.getLogger("notifier.telegram_notifier")
        
        # Track sent notifications
        self.notification_history = []
        self.max_history = 1000
        
        self.logger.info(f"TelegramNotifier initialized with default chat: {default_chat_id}")
    
    async def connect(self) -> bool:
        """Connect to Telegram"""
        return await self.client.connect()
    
    async def disconnect(self):
        """Disconnect from Telegram"""
        await self.client.disconnect()
    
    async def send_notification(self, text: str, chat_id: Optional[str] = None, 
                              is_alert: bool = False) -> bool:
        """
        Send a simple text notification
        
        Args:
            text: Notification text
            chat_id: Target chat ID (uses default if None)
            is_alert: Whether this is an alert (affects notification settings)
            
        Returns:
            True if sent successfully
        """
        target_chat = chat_id or self.default_chat_id
        
        if not target_chat:
            self.logger.error("No chat ID specified for notification")
            return False
        
        try:
            message = TelegramMessage(
                chat_id=target_chat,
                text=text,
                parse_mode="HTML",
                disable_notification=not is_alert
            )
            
            response = await self.client.send_message(message)
            
            # Track notification
            self._add_to_history({
                'type': 'notification',
                'chat_id': target_chat,
                'text': text[:100] + "..." if len(text) > 100 else text,
                'is_alert': is_alert,
                'success': response.get('ok', False),
                'timestamp': datetime.now()
            })
            
            return response.get('ok', False)
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            
            self._add_to_history({
                'type': 'notification',
                'chat_id': target_chat,
                'text': text[:100] + "..." if len(text) > 100 else text,
                'is_alert': is_alert,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            })
            
            return False
    
    async def send_prediction(self, prediction: Prediction, chat_id: Optional[str] = None) -> bool:
        """
        Send a prediction as notification
        
        Args:
            prediction: Prediction object
            chat_id: Target chat ID (uses default if None)
            
        Returns:
            True if sent successfully
        """
        target_chat = chat_id or self.default_chat_id
        
        if not target_chat:
            self.logger.error("No chat ID specified for prediction")
            return False
        
        try:
            success = await self.client.send_prediction(prediction, target_chat)
            
            # Track notification
            self._add_to_history({
                'type': 'prediction',
                'chat_id': target_chat,
                'prediction_id': prediction.id,
                'game': f"{prediction.game.home_team.name} vs {prediction.game.away_team.name}",
                'probability': prediction.probability_metrics.probability_over_25,
                'success': success,
                'timestamp': datetime.now()
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send prediction: {e}")
            
            self._add_to_history({
                'type': 'prediction',
                'chat_id': target_chat,
                'prediction_id': prediction.id,
                'game': f"{prediction.game.home_team.name} vs {prediction.game.away_team.name}",
                'probability': prediction.probability_metrics.probability_over_25,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            })
            
            return False
    
    async def send_alert(self, prediction: Prediction, chat_id: Optional[str] = None) -> bool:
        """
        Send a high-priority alert
        
        Args:
            prediction: Prediction object
            chat_id: Target chat ID (uses default if None)
            
        Returns:
            True if sent successfully
        """
        target_chat = chat_id or self.default_chat_id
        
        if not target_chat:
            self.logger.error("No chat ID specified for alert")
            return False
        
        try:
            success = await self.client.send_alert(prediction, target_chat)
            
            # Track alert
            self._add_to_history({
                'type': 'alert',
                'chat_id': target_chat,
                'prediction_id': prediction.id,
                'game': f"{prediction.game.home_team.name} vs {prediction.game.away_team.name}",
                'probability': prediction.probability_metrics.probability_over_25,
                'priority': prediction.alert_priority,
                'success': success,
                'timestamp': datetime.now()
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
            
            self._add_to_history({
                'type': 'alert',
                'chat_id': target_chat,
                'prediction_id': prediction.id,
                'game': f"{prediction.game.home_team.name} vs {prediction.game.away_team.name}",
                'probability': prediction.probability_metrics.probability_over_25,
                'priority': prediction.alert_priority,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            })
            
            return False
    
    async def send_batch(self, batch: BatchPrediction, chat_id: Optional[str] = None, 
                        include_all: bool = False) -> bool:
        """
        Send batch of predictions
        
        Args:
            batch: BatchPrediction object
            chat_id: Target chat ID (uses default if None)
            include_all: Send all predictions or just high priority
            
        Returns:
            True if sent successfully
        """
        target_chat = chat_id or self.default_chat_id
        
        if not target_chat:
            self.logger.error("No chat ID specified for batch")
            return False
        
        try:
            success = await self.client.send_batch_predictions(batch, target_chat, include_all)
            
            # Track batch
            self._add_to_history({
                'type': 'batch',
                'chat_id': target_chat,
                'batch_id': batch.batch_id,
                'total_predictions': batch.total_games,
                'high_confidence': batch.high_confidence_predictions,
                'include_all': include_all,
                'success': success,
                'timestamp': datetime.now()
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send batch: {e}")
            
            self._add_to_history({
                'type': 'batch',
                'chat_id': target_chat,
                'batch_id': batch.batch_id,
                'total_predictions': batch.total_games,
                'high_confidence': batch.high_confidence_predictions,
                'include_all': include_all,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            })
            
            return False
    
    async def broadcast_alerts(self, predictions: List[Prediction], max_alerts: int = 5) -> Dict[str, Any]:
        """
        Broadcast alerts to all subscribed chats
        
        Args:
            predictions: List of Prediction objects
            max_alerts: Maximum alerts per chat
            
        Returns:
            Results dictionary
        """
        try:
            results = await self.client.send_bulk_alerts(predictions, self.default_chat_id, max_alerts)
            
            # Track broadcast
            self._add_to_history({
                'type': 'broadcast',
                'total_predictions': len(predictions),
                'alerts_sent': results['sent'],
                'alerts_failed': results['failed'],
                'alerts_skipped': results['skipped'],
                'timestamp': datetime.now()
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast alerts: {e}")
            
            self._add_to_history({
                'type': 'broadcast',
                'total_predictions': len(predictions),
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            })
            
            return {
                'total': len(predictions),
                'sent': 0,
                'failed': len(predictions),
                'skipped': 0,
                'details': []
            }
    
    def _add_to_history(self, entry: Dict[str, Any]):
        """Add entry to notification history"""
        self.notification_history.append(entry)
        
        # Limit history size
        if len(self.notification_history) > self.max_history:
            self.notification_history = self.notification_history[-self.max_history:]
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        now = datetime.now()
        
        # Recent notifications (last hour)
        recent = [
            n for n in self.notification_history
            if (now - n['timestamp']).seconds < 3600
        ]
        
        # Count by type
        by_type = {}
        for notification in recent:
            n_type = notification.get('type', 'unknown')
            by_type[n_type] = by_type.get(n_type, 0) + 1
        
        # Success rate
        successful = [n for n in recent if n.get('success', False)]
        success_rate = len(successful) / len(recent) if recent else 1.0
        
        return {
            'total_history': len(self.notification_history),
            'recent_hour': len(recent),
            'by_type': by_type,
            'success_rate_hour': success_rate,
            'default_chat_id': self.default_chat_id,
            'client_connected': self.client.is_connected,
            'client_stats': self.client.get_statistics()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        client_health = await self.client.health_check()
        
        return {
            'notifier': {
                'default_chat_id': self.default_chat_id,
                'notification_history': len(self.notification_history),
                'status': 'healthy' if client_health.get('status') == 'healthy' else 'unhealthy'
            },
            'client': client_health,
            'timestamp': datetime.now().isoformat()
        }


# Utility functions

async def test_telegram_connection(bot_token: str, chat_id: str) -> Dict[str, Any]:
    """
    Test Telegram connection and send a test message
    
    Args:
        bot_token: Telegram Bot API token
        chat_id: Chat ID to send test message to
        
    Returns:
        Test results dictionary
    """
    results = {
        'connection': False,
        'message_sent': False,
        'bot_info': None,
        'error': None,
        'timestamp': datetime.now().isoformat()
    }
    
    client = None
    
    try:
        # Create client
        client = create_telegram_client(bot_token)
        
        # Connect
        connected = await client.connect()
        results['connection'] = connected
        
        if connected:
            results['bot_info'] = client.bot_info
            
            # Send test message
            test_message = TelegramMessage(
                chat_id=chat_id,
                text="✅ <b>Test Message</b>\n\nTelegram connection test successful!\n\nBot is online and ready to send predictions.",
                parse_mode="HTML"
            )
            
            response = await client.send_message(test_message)
            results['message_sent'] = response.get('ok', False)
            
            if not results['message_sent']:
                results['error'] = response.get('description', 'Failed to send message')
        
    except Exception as e:
        results['error'] = str(e)
        results['connection'] = False
        
    finally:
        if client:
            await client.disconnect()
    
    return results


def format_error_message(error: Exception, context: str = "") -> str:
    """
    Format error message for Telegram
    
    Args:
        error: Exception object
        context: Context description
        
    Returns:
        Formatted error message
    """
    base_message = f"❌ <b>Error{f' in {context}' if context else ''}</b>\n\n"
    
    if isinstance(error, TelegramError):
        return base_message + f"Telegram API Error: {error.message}\n\nCode: {error.error_code}"
    else:
        return base_message + f"Error: {str(error)}\n\nPlease check the logs for details."


# Example usage and testing

async def main():
    """Test the Telegram client"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Get credentials from environment
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    if not BOT_TOKEN or not CHAT_ID:
        print("Error: Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        return
    
    print("Testing Telegram client...")
    
    # Test connection
    print("\n1. Testing connection...")
    test_results = await test_telegram_connection(BOT_TOKEN, CHAT_ID)
    
    if test_results['connection'] and test_results['message_sent']:
        print("✅ Connection test passed!")
        print(f"   Bot: @{test_results['bot_info']['result']['username']}")
    else:
        print(f"❌ Connection test failed: {test_results.get('error', 'Unknown error')}")
        return
    
    # Create notifier
    print("\n2. Creating notifier...")
    notifier = TelegramNotifier(BOT_TOKEN, CHAT_ID)
    
    # Connect
    connected = await notifier.connect()
    if not connected:
        print("❌ Failed to connect notifier")
        return
    
    print("✅ Notifier connected")
    
    # Send test notification
    print("\n3. Sending test notification...")
    success = await notifier.send_notification(
        "🧪 <b>Test Notification</b>\n\nThis is a test notification from the Over/Under Predictor system.\n\nSystem is online and ready to send predictions!",
        is_alert=True
    )
    
    if success:
        print("✅ Test notification sent successfully!")
    else:
        print("❌ Failed to send test notification")
    
    # Get statistics
    print("\n4. Getting statistics...")
    stats = notifier.get_notification_stats()
    print(f"   Recent notifications: {stats['recent_hour']}")
    print(f"   Success rate: {stats['success_rate_hour']:.1%}")
    
    # Health check
    print("\n5. Running health check...")
    health = await notifier.health_check()
    print(f"   Status: {health['notifier']['status']}")
    
    # Disconnect
    print("\n6. Disconnecting...")
    await notifier.disconnect()
    print("✅ Disconnected successfully")
    
    print("\n" + "="*50)
    print("Telegram client test completed!")
    print("="*50)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
