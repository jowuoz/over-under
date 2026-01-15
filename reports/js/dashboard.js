// reports/js/dashboard.js

// Dashboard JavaScript
console.log('📊 Dashboard JavaScript loaded');

// Global variables
let countdownInterval;
let confidenceChart = null;
let leagueChart = null;
let liveGamesCount = 0;
let currentPredictions = [];

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    
    // Start countdown timer
    startCountdown();
    
    // Load all dashboard data
    loadDashboardData();
    
    // Initialize charts (they will be populated when data loads)
    initializeCharts();
    
    // Set up auto-refresh (every 5 minutes)
    setInterval(loadDashboardData, 300000);
    
    // Initialize any interactive elements
    initializeInteractiveElements();
});

// Start countdown timer
function startCountdown() {
    console.log('Starting countdown timer');
    
    let seconds = 300; // 5 minutes
    const countdownElement = document.getElementById('countdown');
    
    if (!countdownElement) {
        console.warn('Countdown element not found');
        return;
    }
    
    // Clear any existing interval
    if (countdownInterval) {
        clearInterval(countdownInterval);
    }
    
    countdownInterval = setInterval(() => {
        seconds--;
        
        if (countdownElement) {
            countdownElement.textContent = seconds;
            
            // Add warning color when less than 60 seconds
            if (seconds <= 60) {
                countdownElement.style.color = '#ef4444';
                countdownElement.classList.add('pulse');
            } else if (seconds <= 120) {
                countdownElement.style.color = '#f59e0b';
            } else {
                countdownElement.style.color = '';
                countdownElement.classList.remove('pulse');
            }
        }
        
        if (seconds <= 0) {
            clearInterval(countdownInterval);
            showNotification('Refreshing dashboard...', 'info');
            setTimeout(() => {
                location.reload();
            }, 1000);
        }
    }, 1000);
}

// Load all dashboard data
function loadDashboardData() {
    console.log('Loading dashboard data...');
    
    // Update last updated time
    updateLastUpdated();
    
    // Load main data
    loadMainData();
    
    // Load charts data
    loadChartsData();
    
    // Load live games
    loadLiveGames();
    
    // Load recent alerts
    loadRecentAlerts();
}

// Load main dashboard data
function loadMainData() {
    console.log('Loading main data from latest.json...');
    
    fetch('./data/latest.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Main data loaded:', data);
            
            // Store predictions for filtering/search
            currentPredictions = data.predictions || [];
            
            // Update dashboard stats
            updateDashboardStats(data);
            
            // Update games grid
            updateGamesGrid(data);
        })
        .catch(error => {
            console.error('Error loading main data:', error);
            showNotification('Could not load dashboard data', 'warning');
            
            // Use fallback sample data
            useSampleData();
        });
}

// Update dashboard statistics
function updateDashboardStats(data) {
    console.log('Updating dashboard stats');
    
    // Calculate live games count
    const liveGames = (data.predictions || []).filter(p => 
        p.status && ['live', 'LIVE', 'IN_PLAY'].includes(p.status)
    );
    liveGamesCount = liveGames.length;
    
    // Update live games count
    const liveGamesElement = document.querySelector('.stat-number');
    if (liveGamesElement) {
        liveGamesElement.textContent = liveGamesCount;
    }
    
    // Update active alerts
    const alertsCount = data.alerts ? data.alerts.length : 0;
    const activeAlertsElement = document.querySelectorAll('.stat-number')[1];
    if (activeAlertsElement) {
        activeAlertsElement.textContent = alertsCount;
    }
    
    // Update average confidence
    const avgConfidence = data.avg_confidence || calculateAverageConfidence(data.predictions || []);
    const avgConfidenceElement = document.querySelectorAll('.stat-number')[2];
    if (avgConfidenceElement) {
        avgConfidenceElement.textContent = avgConfidence.toFixed(1) + '%';
    }
    
    // Update top league
    const topLeague = findTopLeague(data.predictions || []);
    const topLeagueElement = document.querySelectorAll('.stat-number')[3];
    if (topLeagueElement) {
        topLeagueElement.textContent = topLeague;
    }
}

// Calculate average confidence from predictions
function calculateAverageConfidence(predictions) {
    if (!predictions || predictions.length === 0) return 0;
    
    const validConfidences = predictions
        .filter(p => p.confidence != null)
        .map(p => p.confidence * 100);
    
    if (validConfidences.length === 0) return 0;
    
    const sum = validConfidences.reduce((a, b) => a + b, 0);
    return sum / validConfidences.length;
}

// Find the league with most games
function findTopLeague(predictions) {
    if (!predictions || predictions.length === 0) return 'N/A';
    
    const leagueCounts = {};
    predictions.forEach(prediction => {
        const league = prediction.league || 'Unknown';
        leagueCounts[league] = (leagueCounts[league] || 0) + 1;
    });
    
    let topLeague = 'N/A';
    let maxCount = 0;
    
    for (const [league, count] of Object.entries(leagueCounts)) {
        if (count > maxCount) {
            maxCount = count;
            topLeague = league;
        }
    }
    
    return topLeague;
}

// Update games grid
function updateGamesGrid(data) {
    console.log('Updating games grid');
    
    const gamesGrid = document.querySelector('.games-grid');
    if (!gamesGrid) return;
    
    const liveGames = (data.predictions || []).filter(p => 
        p.status && ['live', 'LIVE', 'IN_PLAY'].includes(p.status)
    );
    
    if (liveGames.length === 0) {
        gamesGrid.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-clock"></i>
                <p>No live games being analyzed</p>
                <p class="small-text">Check back during match hours</p>
            </div>
        `;
        return;
    }
    
    // Sort by probability (highest first)
    const sortedGames = liveGames.sort((a, b) => 
        (b.over_25_prob || 0) - (a.over_25_prob || 0)
    );
    
    // Take top 6 games
    const topGames = sortedGames.slice(0, 6);
    
    let html = '';
    topGames.forEach(game => {
        html += formatGameCard(game);
    });
    
    gamesGrid.innerHTML = html;
    
    // Add click handlers to details buttons
    document.querySelectorAll('.btn-details').forEach(button => {
        button.addEventListener('click', function() {
            const gameId = this.getAttribute('data-game-id');
            viewGameDetails(gameId);
        });
    });
}

// Format game card HTML
function formatGameCard(game) {
    const homeTeam = game.home_team || 'Home';
    const awayTeam = game.away_team || 'Away';
    const homeScore = game.home_score || 0;
    const awayScore = game.away_score || 0;
    const totalGoals = homeScore + awayScore;
    const minute = game.minute || 0;
    const league = game.league || 'Unknown League';
    const probability = (game.over_25_prob || 0) * 100;
    const confidence = (game.confidence || 0) * 100;
    const gameId = game.id || Math.random().toString(36).substr(2, 9);
    
    // Determine status
    let statusClass, statusText, statusIcon;
    if (totalGoals >= 3) {
        statusClass = 'status-over';
        statusText = 'OVER 2.5';
        statusIcon = 'fas fa-arrow-up';
    } else if (totalGoals === 2 && minute >= 70) {
        statusClass = 'status-close';
        statusText = 'CLOSE';
        statusIcon = 'fas fa-hourglass-half';
    } else {
        statusClass = 'status-under';
        statusText = 'UNDER 2.5';
        statusIcon = 'fas fa-arrow-down';
    }
    
    // League logo (placeholder)
    const logoUrl = `https://img.icons8.com/color/96/000000/football2--v1.png`;
    
    return `
        <div class="game-card">
            <div class="game-header">
                <img src="${logoUrl}" alt="${league}" class="league-logo">
                <span class="league-name">${league}</span>
                <span class="game-minute">${minute}'</span>
            </div>
            <div class="game-teams">
                <div class="team home-team">
                    <span class="team-name">${homeTeam}</span>
                    <span class="team-score">${homeScore}</span>
                </div>
                <div class="vs-separator">VS</div>
                <div class="team away-team">
                    <span class="team-score">${awayScore}</span>
                    <span class="team-name">${awayTeam}</span>
                </div>
            </div>
            <div class="game-status">
                <div class="status-indicator ${statusClass}">
                    <i class="${statusIcon}"></i>
                    <span>${statusText}</span>
                </div>
                <div class="game-probability">
                    <i class="fas fa-chart-line"></i>
                    <span>Over 2.5: ${probability.toFixed(1)}%</span>
                </div>
            </div>
            <div class="game-footer">
                <span class="game-updated">
                    <i class="fas fa-sync-alt"></i> Updated ${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                </span>
                <button class="btn-details" data-game-id="${gameId}">
                    Details <i class="fas fa-chevron-right"></i>
                </button>
            </div>
        </div>
    `;
}

// Initialize charts
function initializeCharts() {
    console.log('Initializing charts');
    
    // Wait a bit for DOM to be fully ready
    setTimeout(() => {
        loadConfidenceChart();
        loadLeagueChart();
    }, 500);
}

// Load confidence chart
function loadConfidenceChart() {
    console.log('Loading confidence chart...');
    
    fetch('./data/confidence_chart.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('Confidence chart data not found');
            }
            return response.json();
        })
        .then(data => {
            console.log('Confidence chart data:', data);
            renderConfidenceChart(data);
        })
        .catch(error => {
            console.error('Error loading confidence chart:', error);
            createSampleConfidenceChart();
        });
}

// Render confidence chart
function renderConfidenceChart(chartData) {
    const ctx = document.getElementById('confidenceChart');
    if (!ctx) {
        console.warn('Confidence chart canvas not found');
        return;
    }
    
    // Destroy existing chart if it exists
    if (confidenceChart) {
        confidenceChart.destroy();
    }
    
    confidenceChart = new Chart(ctx.getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: chartData.labels || ['Low', 'Medium', 'High', 'Very High'],
            datasets: [{
                data: chartData.data || [10, 20, 30, 40],
                backgroundColor: chartData.colors || ['#ef4444', '#f59e0b', '#10b981', '#3b82f6'],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: {
                            size: 11
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((value / total) * 100);
                            return `${label}: ${value} predictions (${percentage}%)`;
                        }
                    }
                }
            },
            cutout: '65%'
        }
    });
}

// Load league distribution chart
function loadLeagueChart() {
    console.log('Loading league chart...');
    
    fetch('./data/league_chart.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('League chart data not found');
            }
            return response.json();
        })
        .then(data => {
            console.log('League chart data:', data);
            renderLeagueChart(data);
        })
        .catch(error => {
            console.error('Error loading league chart:', error);
            createSampleLeagueChart();
        });
}

// Render league chart
function renderLeagueChart(chartData) {
    const ctx = document.getElementById('leagueChart');
    if (!ctx) {
        console.warn('League chart canvas not found');
        return;
    }
    
    // Destroy existing chart if it exists
    if (leagueChart) {
        leagueChart.destroy();
    }
    
    leagueChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: chartData.labels || ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'],
            datasets: [{
                label: 'Number of Games',
                data: chartData.data || [12, 8, 6, 5, 4],
                backgroundColor: chartData.colors || [
                    '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe'
                ],
                borderColor: chartData.colors || [
                    '#1d4ed8', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1,
                        precision: 0
                    },
                    grid: {
                        drawBorder: false
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// Load live games
function loadLiveGames() {
    console.log('Loading live games...');
    
    // This would typically make an API call
    // For now, we'll rely on the data from latest.json
    updateLiveGamesBadge();
}

// Update live games badge
function updateLiveGamesBadge() {
    const badge = document.querySelector('.badge-live');
    if (badge) {
        badge.textContent = `${liveGamesCount} Live`;
        
        // Add animation if games are live
        if (liveGamesCount > 0) {
            badge.classList.add('pulse');
        } else {
            badge.classList.remove('pulse');
        }
    }
}

// Load recent alerts
function loadRecentAlerts() {
    console.log('Loading recent alerts...');
    
    fetch('./data/latest.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('Could not load alerts');
            }
            return response.json();
        })
        .then(data => {
            const alerts = data.alerts || [];
            updateRecentAlerts(alerts.slice(-10)); // Last 10 alerts
        })
        .catch(error => {
            console.error('Error loading alerts:', error);
        });
}

// Update recent alerts section
function updateRecentAlerts(alerts) {
    console.log('Updating recent alerts:', alerts.length);
    
    const alertsContainer = document.querySelector('.alerts-container');
    if (!alertsContainer) return;
    
    if (alerts.length === 0) {
        alertsContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-bell-slash"></i>
                <p>No active alerts at the moment</p>
                <p class="small-text">New alerts will appear here automatically</p>
            </div>
        `;
        return;
    }
    
    // Sort by probability (highest first)
    const sortedAlerts = alerts.sort((a, b) => 
        (b.over_25_prob || 0) - (a.over_25_prob || 0)
    );
    
    let html = '';
    sortedAlerts.forEach(alert => {
        html += formatAlertCard(alert);
    });
    
    alertsContainer.innerHTML = html;
    
    // Add click handlers to alert buttons
    document.querySelectorAll('.alert-card .btn-action').forEach(button => {
        button.addEventListener('click', function() {
            const action = this.getAttribute('data-action');
            const alertId = this.getAttribute('data-alert-id');
            
            if (action === 'track') {
                trackAlert(alertId);
            } else if (action === 'share') {
                const homeTeam = this.getAttribute('data-home-team');
                const awayTeam = this.getAttribute('data-away-team');
                const probability = this.getAttribute('data-probability');
                shareAlert(homeTeam, awayTeam, probability);
            }
        });
    });
}

// Format alert card HTML
function formatAlertCard(alert) {
    const probability = (alert.over_25_prob || 0) * 100;
    const confidence = (alert.confidence || 0) * 100;
    const homeTeam = alert.home_team || 'Unknown';
    const awayTeam = alert.away_team || 'Unknown';
    const homeScore = alert.home_score || 0;
    const awayScore = alert.away_score || 0;
    const minute = alert.minute || 0;
    const league = alert.league || 'Unknown League';
    const alertId = alert.id || Math.random().toString(36).substr(2, 9);
    
    // Determine badge color
    let badgeClass, badgeIcon;
    if (probability >= 80) {
        badgeClass = 'badge-high';
        badgeIcon = 'fas fa-fire';
    } else if (probability >= 70) {
        badgeClass = 'badge-medium';
        badgeIcon = 'fas fa-exclamation-triangle';
    } else {
        badgeClass = 'badge-low';
        badgeIcon = 'fas fa-info-circle';
    }
    
    return `
        <div class="alert-card">
            <div class="alert-header">
                <div class="alert-badge ${badgeClass}">
                    <i class="${badgeIcon}"></i>
                    <span>Over 2.5: ${probability.toFixed(1)}%</span>
                </div>
                <div class="alert-time">
                    <i class="fas fa-clock"></i> ${minute}'
                </div>
            </div>
            <div class="alert-content">
                <h4>${homeTeam} vs ${awayTeam}</h4>
                <div class="alert-score">
                    <span class="score">${homeScore} - ${awayScore}</span>
                </div>
                <div class="alert-details">
                    <div class="detail-item">
                        <i class="fas fa-bullseye"></i>
                        <span>Confidence: ${confidence.toFixed(1)}%</span>
                    </div>
                    <div class="detail-item">
                        <i class="fas fa-trophy"></i>
                        <span>${league}</span>
                    </div>
                </div>
            </div>
            <div class="alert-actions">
                <button class="btn-action" data-action="track" data-alert-id="${alertId}">
                    <i class="fas fa-eye"></i> Track
                </button>
                <button class="btn-action" data-action="share" 
                        data-alert-id="${alertId}"
                        data-home-team="${homeTeam}"
                        data-away-team="${awayTeam}"
                        data-probability="${probability.toFixed(1)}">
                    <i class="fas fa-share-alt"></i> Share
                </button>
            </div>
        </div>
    `;
}

// Initialize interactive elements
function initializeInteractiveElements() {
    console.log('Initializing interactive elements');
    
    // View all alerts button
    const viewAllBtn = document.querySelector('.btn-view-all');
    if (viewAllBtn) {
        viewAllBtn.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Navigating to alerts page');
            window.location.href = 'alerts.html';
        });
    }
    
    // Statistics link
    const statsLinks = document.querySelectorAll('a[href="statistics.html"]');
    statsLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Navigating to statistics page');
            window.location.href = 'statistics.html';
        });
    });
    
    // Performance link
    const perfLinks = document.querySelectorAll('a[href="performance.html"]');
    perfLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Navigating to performance page');
            window.location.href = 'performance.html';
        });
    });
    
    // Leagues link
    const leagueLinks = document.querySelectorAll('a[href="leagues.html"]');
    leagueLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Navigating to leagues page');
            window.location.href = 'leagues.html';
        });
    });
    
    // Add manual refresh button if needed
    addManualRefreshButton();
}

// Add manual refresh button
function addManualRefreshButton() {
    const headerStats = document.querySelector('.header-stats');
    if (headerStats && !document.getElementById('manualRefreshBtn')) {
        const refreshBtn = document.createElement('div');
        refreshBtn.className = 'stat-badge';
        refreshBtn.id = 'manualRefreshBtn';
        refreshBtn.innerHTML = `
            <i class="fas fa-sync-alt"></i>
            <span>Refresh Now</span>
        `;
        refreshBtn.style.cursor = 'pointer';
        refreshBtn.addEventListener('click', function() {
            refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Refreshing...</span>';
            refreshBtn.style.opacity = '0.7';
            
            loadDashboardData();
            
            setTimeout(() => {
                refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i><span>Refresh Now</span>';
                refreshBtn.style.opacity = '1';
                showNotification('Dashboard refreshed successfully!', 'success');
            }, 1000);
        });
        
        headerStats.appendChild(refreshBtn);
    }
}

// Update last updated time
function updateLastUpdated() {
    const lastUpdatedElements = document.querySelectorAll('.stat-badge');
    if (lastUpdatedElements.length >= 1) {
        const timeString = new Date().toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit'
        });
        lastUpdatedElements[0].querySelector('span').textContent = `Last updated: ${timeString}`;
    }
}

// Track alert
function trackAlert(alertId) {
    console.log('Tracking alert:', alertId);
    
    // In a real implementation, this would send tracking data to your backend
    showNotification(`Now tracking alert #${alertId}`, 'info');
    
    // Simulate tracking action
    const alertElement = document.querySelector(`[data-alert-id="${alertId}"]`);
    if (alertElement) {
        alertElement.innerHTML = '<i class="fas fa-check"></i> Tracking';
        alertElement.style.background = '#10b981';
        
        setTimeout(() => {
            alertElement.innerHTML = '<i class="fas fa-eye"></i> Track';
            alertElement.style.background = '';
        }, 2000);
    }
}

// Share alert
function shareAlert(homeTeam, awayTeam, probability) {
    console.log('Sharing alert:', homeTeam, awayTeam, probability);
    
    const text = `⚽ Alert: ${homeTeam} vs ${awayTeam} - Over 2.5 probability: ${probability}%\n\nCheck out the Over/Under Predictor dashboard for more insights!`;
    
    if (navigator.share) {
        // Use Web Share API if available
        navigator.share({
            title: 'Over/Under Prediction Alert',
            text: text,
            url: window.location.href
        })
        .then(() => {
            console.log('Alert shared successfully');
            showNotification('Alert shared!', 'success');
        })
        .catch(error => {
            console.error('Error sharing:', error);
            copyToClipboard(text);
        });
    } else {
        // Fallback to clipboard
        copyToClipboard(text);
    }
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text)
        .then(() => {
            console.log('Alert copied to clipboard');
            showNotification('Alert copied to clipboard!', 'success');
        })
        .catch(error => {
            console.error('Clipboard error:', error);
            showNotification('Could not copy to clipboard', 'warning');
        });
}

// View game details
function viewGameDetails(gameId) {
    console.log('Viewing game details:', gameId);
    
    // In a real implementation, this would open a modal or navigate to a details page
    showNotification(`Viewing details for game #${gameId}`, 'info');
    
    // Find the game in current predictions
    const game = currentPredictions.find(p => p.id === gameId);
    
    if (game) {
        // Create and show a simple modal with game details
        showGameDetailsModal(game);
    } else {
        showNotification('Game details not available', 'warning');
    }
}

// Show game details modal
function showGameDetailsModal(game) {
    const homeTeam = game.home_team || 'Home';
    const awayTeam = game.away_team || 'Away';
    const homeScore = game.home_score || 0;
    const awayScore = game.away_score || 0;
    const minute = game.minute || 0;
    const league = game.league || 'Unknown League';
    const probability = (game.over_25_prob || 0) * 100;
    const confidence = (game.confidence || 0) * 100;
    const status = game.status || 'unknown';
    
    // Create modal HTML
    const modalHtml = `
        <div class="modal-overlay" id="gameDetailsModal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3><i class="fas fa-info-circle"></i> Game Details</h3>
                    <button class="modal-close" onclick="closeModal()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="game-details-header">
                        <h4>${homeTeam} vs ${awayTeam}</h4>
                        <div class="game-score-large">
                            ${homeScore} - ${awayScore}
                        </div>
                        <div class="game-minute-large">
                            <i class="fas fa-clock"></i> ${minute}'
                        </div>
                    </div>
                    
                    <div class="details-grid">
                        <div class="detail-item">
                            <i class="fas fa-trophy"></i>
                            <span><strong>League:</strong> ${league}</span>
                        </div>
                        <div class="detail-item">
                            <i class="fas fa-chart-line"></i>
                            <span><strong>Over 2.5 Probability:</strong> ${probability.toFixed(1)}%</span>
                        </div>
                        <div class="detail-item">
                            <i class="fas fa-bullseye"></i>
                            <span><strong>Confidence:</strong> ${confidence.toFixed(1)}%</span>
                        </div>
                        <div class="detail-item">
                            <i class="fas fa-play-circle"></i>
                            <span><strong>Status:</strong> ${status.toUpperCase()}</span>
                        </div>
                    </div>
                    
                    <div class="prediction-analysis">
                        <h5><i class="fas fa-chart-bar"></i> Prediction Analysis</h5>
                        <div class="probability-meter">
                            <div class="meter-label">
                                <span>Under 2.5</span>
                                <span>Over 2.5</span>
                            </div>
                            <div class="meter-bar">
                                <div class="meter-fill" style="width: ${100 - probability}%"></div>
                            </div>
                            <div class="meter-value">
                                <span>${(100 - probability).toFixed(1)}%</span>
                                <span>${probability.toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn-secondary" onclick="closeModal()">
                        Close
                    </button>
                    <button class="btn-primary" onclick="shareAlert('${homeTeam}', '${awayTeam}', '${probability.toFixed(1)}')">
                        <i class="fas fa-share-alt"></i> Share Prediction
                    </button>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to page
    const modalContainer = document.createElement('div');
    modalContainer.innerHTML = modalHtml;
    document.body.appendChild(modalContainer.firstElementChild);
    
    // Add modal styles if not already present
    addModalStyles();
    
    // Prevent scrolling on body
    document.body.style.overflow = 'hidden';
}

// Close modal
function closeModal() {
    const modal = document.getElementById('gameDetailsModal');
    if (modal) {
        modal.remove();
        document.body.style.overflow = '';
    }
}

// Add modal styles
function addModalStyles() {
    if (document.getElementById('modal-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'modal-styles';
    style.textContent = `
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            animation: fadeIn 0.3s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .modal-content {
            background: white;
            border-radius: 1rem;
            width: 90%;
            max-width: 500px;
            max-height: 90vh;
            overflow-y: auto;
            animation: slideUp 0.3s ease-out;
        }
        
        @keyframes slideUp {
            from { 
                transform: translateY(50px);
                opacity: 0;
            }
            to { 
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.5rem;
            border-bottom: 2px solid #f1f5f9;
        }
        
        .modal-header h3 {
            font-size: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .modal-close {
            background: none;
            border: none;
            font-size: 1.25rem;
            color: #64748b;
            cursor: pointer;
            padding: 0.5rem;
        }
        
        .modal-close:hover {
            color: #1e293b;
        }
        
        .modal-body {
            padding: 1.5rem;
        }
        
        .game-details-header {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        .game-score-large {
            font-size: 3rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .game-minute-large {
            font-size: 1.1rem;
            color: #64748b;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .details-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .detail-item {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.75rem;
            background: #f8fafc;
            border-radius: 0.5rem;
        }
        
        .prediction-analysis {
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 2px solid #f1f5f9;
        }
        
        .probability-meter {
            margin-top: 1rem;
        }
        
        .meter-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 0.5rem;
        }
        
        .meter-bar {
            height: 10px;
            background: #e2e8f0;
            border-radius: 5px;
            overflow: hidden;
        }
        
        .meter-fill {
            height: 100%;
            background: linear-gradient(90deg, #ef4444 0%, #10b981 100%);
        }
        
        .meter-value {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        
        .modal-footer {
            display: flex;
            gap: 1rem;
            padding: 1.5rem;
            border-top: 2px solid #f1f5f9;
        }
        
        .btn-primary, .btn-secondary {
            flex: 1;
            padding: 0.75rem;
            border-radius: 0.5rem;
            border: none;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .btn-primary {
            background: #3b82f6;
            color: white;
        }
        
        .btn-primary:hover {
            background: #2563eb;
        }
        
        .btn-secondary {
            background: #f1f5f9;
            color: #475569;
        }
        
        .btn-secondary:hover {
            background: #e2e8f0;
        }
        
        .small-text {
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 0.5rem;
        }
    `;
    
    document.head.appendChild(style);
}

// Show notification
function showNotification(message, type = 'info') {
    console.log(`Notification [${type}]:`, message);
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${getNotificationIcon(type)}"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add styles for notification if not already present
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            .notification-container {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
            }
            
            .notification {
                background: white;
                padding: 1rem 1.5rem;
                border-radius: 0.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                display: flex;
                align-items: center;
                gap: 1rem;
                margin-bottom: 0.5rem;
                animation: slideIn 0.3s ease-out;
                border-left: 4px solid #3b82f6;
            }
            
            .notification-success {
                border-left-color: #10b981;
            }
            
            .notification-info {
                border-left-color: #3b82f6;
            }
            
            .notification-warning {
                border-left-color: #f59e0b;
            }
            
            .notification button {
                background: none;
                border: none;
                cursor: pointer;
                color: #64748b;
                margin-left: auto;
            }
            
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    // Get or create notification container
    let container = document.querySelector('.notification-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'notification-container';
        document.body.appendChild(container);
    }
    
    container.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Get notification icon based on type
function getNotificationIcon(type) {
    switch(type) {
        case 'success': return 'check-circle';
        case 'warning': return 'exclamation-triangle';
        case 'error': return 'exclamation-circle';
        default: return 'info-circle';
    }
}

// Create sample confidence chart (fallback)
function createSampleConfidenceChart() {
    console.log('Creating sample confidence chart');
    
    const sampleData = {
        labels: ['Low (<50%)', 'Medium (50-70%)', 'High (70-85%)', 'Very High (>85%)'],
        data: [15, 25, 35, 25],
        colors: ['#ef4444', '#f59e0b', '#10b981', '#3b82f6']
    };
    
    renderConfidenceChart(sampleData);
}

// Create sample league chart (fallback)
function createSampleLeagueChart() {
    console.log('Creating sample league chart');
    
    const sampleData = {
        labels: ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'],
        data: [12, 8, 6, 5, 4],
        colors: ['#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe']
    };
    
    renderLeagueChart(sampleData);
}

// Use sample data when real data is unavailable
function useSampleData() {
    console.log('Using sample data');
    
    const sampleData = {
        predictions: [
            {
                id: '1',
                home_team: 'Manchester City',
                away_team: 'Liverpool',
                league: 'Premier League',
                home_score: 2,
                away_score: 1,
                minute: 75,
                over_25_prob: 0.85,
                confidence: 0.92,
                status: 'live'
            },
            {
                id: '2',
                home_team: 'Real Madrid',
                away_team: 'Barcelona',
                league: 'La Liga',
                home_score: 1,
                away_score: 1,
                minute: 60,
                over_25_prob: 0.72,
                confidence: 0.78,
                status: 'live'
            }
        ],
        alerts: [
            {
                id: 'alert1',
                home_team: 'Arsenal',
                away_team: 'Chelsea',
                league: 'Premier League',
                home_score: 1,
                away_score: 0,
                minute: 55,
                over_25_prob: 0.82,
                confidence: 0.88,
                status: 'live'
            }
        ],
        avg_confidence: 78.5
    };
    
    updateDashboardStats(sampleData);
    updateGamesGrid(sampleData);
    updateRecentAlerts(sampleData.alerts);
    
    showNotification('Using sample data for demonstration', 'info');
}

// Load charts data (from multiple sources)
function loadChartsData() {
    console.log('Loading charts data');
    
    // Charts are loaded separately by loadConfidenceChart() and loadLeagueChart()
    // This function can be used to load additional chart data if needed
}

// Make functions available globally
window.startCountdown = startCountdown;
window.loadConfidenceChart = loadConfidenceChart;
window.loadLeagueChart = loadLeagueChart;
window.trackAlert = trackAlert;
window.shareAlert = shareAlert;
window.viewGameDetails = viewGameDetails;
window.closeModal = closeModal;
window.showNotification = showNotification;
