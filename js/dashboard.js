
// Dashboard JavaScript
function startCountdown() {
    let seconds = 300;
    const countdownElement = document.getElementById('countdown');
    
    const interval = setInterval(() => {
        seconds--;
        if (countdownElement) {
            countdownElement.textContent = seconds;
        }
        
        if (seconds <= 0) {
            clearInterval(interval);
            location.reload();
        }
    }, 1000);
}

function loadConfidenceChart() {
    fetch('data/confidence_chart.json')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('confidenceChart').getContext('2d');
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: data.labels,
                    datasets: [{
                        data: data.data,
                        backgroundColor: data.colors,
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
                                usePointStyle: true
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.label}: ${context.raw} predictions`;
                                }
                            }
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error loading confidence chart:', error));
}

function loadLeagueChart() {
    fetch('data/league_chart.json')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('leagueChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: 'Number of Games',
                        data: data.data,
                        backgroundColor: data.colors,
                        borderWidth: 1,
                        borderColor: 'rgba(0, 0, 0, 0.1)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error loading league chart:', error));
}

function trackAlert(alertId) {
    console.log('Tracking alert:', alertId);
    // Implement alert tracking logic
}

function shareAlert(homeTeam, awayTeam, probability) {
    const text = `⚽ Alert: ${homeTeam} vs ${awayTeam} - Over 2.5 probability: ${probability}%`;
    if (navigator.share) {
        navigator.share({
            title: 'Over/Under Alert',
            text: text,
            url: window.location.href
        });
    } else {
        navigator.clipboard.writeText(text);
        alert('Alert copied to clipboard!');
    }
}

function viewGameDetails(gameId) {
    console.log('Viewing game details:', gameId);
    // Implement game details view
}
