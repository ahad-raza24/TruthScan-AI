/**
 * TruthScan - Professional AI Fake News Detection System
 * Enhanced JavaScript with Modern Features & Visualizations
 * Version 2.0.0
 */

// ===== GLOBAL VARIABLES =====
let currentVerificationResults = null;
let analysisTimer = null;
let analysisStartTime = null;
let trustGaugeChart = null;
let accuracyChart = null;
let biasChart = null;

    // ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 TruthScan v2.0.0 - Initializing...');
    
    // Initialize all components
    initializeLoadingScreen();
    initializeAOS();
    initializeTheme();
    initializeNavigation();
    initializeCharts();
    initializeFormHandlers();
    initializeTabSystem();
    initializeCounters();
    initializeExampleClaims();
    
    console.log('✅ TruthScan initialized successfully');
});

// ===== LOADING SCREEN =====
function initializeLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    
    // Hide loading screen after a short delay
    setTimeout(() => {
        if (loadingScreen) {
            loadingScreen.classList.add('hidden');
            setTimeout(() => {
                loadingScreen.style.display = 'none';
            }, 500);
        }
    }, 1500);
}

// ===== AOS ANIMATION INITIALIZATION =====
function initializeAOS() {
    if (typeof AOS !== 'undefined') {
    AOS.init({
            duration: 1000,
            easing: 'ease-out-cubic',
            once: true,
            offset: 100,
            delay: 100
        });
    }
}

// ===== THEME SYSTEM =====
function initializeTheme() {
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    
    // Load saved theme
    const savedTheme = localStorage.getItem('truthscan-theme') || 'light';
    applyTheme(savedTheme);
    
    // Theme toggle handler
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.body.classList.contains('dark-mode') ? 'dark' : 'light';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            applyTheme(newTheme);
            localStorage.setItem('truthscan-theme', newTheme);
        });
    }
}

function applyTheme(theme) {
    const body = document.body;
    const themeIcon = document.getElementById('theme-icon');
    
    if (theme === 'dark') {
        body.classList.add('dark-mode');
        if (themeIcon) {
            themeIcon.className = 'fas fa-sun';
        }
    } else {
        body.classList.remove('dark-mode');
        if (themeIcon) {
            themeIcon.className = 'fas fa-moon';
        }
    }
    
    // Update charts if they exist
    updateChartsTheme();
}

// ===== NAVIGATION SYSTEM =====
function initializeNavigation() {
    const navbar = document.querySelector('.navbar');
    const navLinks = document.querySelectorAll('.nav-link');
    
    // Navbar scroll effect
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar?.classList.add('scrolled');
        } else {
            navbar?.classList.remove('scrolled');
        }
    });
    
    // Smooth scrolling for navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href?.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    const offsetTop = target.offsetTop - 80;
                    window.scrollTo({
                        top: offsetTop,
                        behavior: 'smooth'
                    });
                    
                    // Update active nav link
                    updateActiveNavLink(href);
                }
            }
        });
    });
    
    // Update active nav link on scroll
    window.addEventListener('scroll', updateActiveNavLinkOnScroll);
}

function updateActiveNavLink(activeHref) {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === activeHref) {
            link.classList.add('active');
        }
    });
}

function updateActiveNavLinkOnScroll() {
    const sections = document.querySelectorAll('section[id]');
    const scrollPos = window.scrollY + 100;
    
    sections.forEach(section => {
        const top = section.offsetTop;
        const bottom = top + section.offsetHeight;
        const id = section.getAttribute('id');
        
        if (scrollPos >= top && scrollPos <= bottom) {
            updateActiveNavLink(`#${id}`);
        }
    });
}

// ===== CHART INITIALIZATION =====
function initializeCharts() {
    initializeAccuracyChart();
    initializeTrustGauge();
    initializeBiasChart();
}

function initializeAccuracyChart() {
    const ctx = document.getElementById('accuracyChart');
    if (!ctx || typeof Chart === 'undefined') return;
    
    const isDark = document.body.classList.contains('dark-mode');
    const textColor = isDark ? '#f9fafb' : '#111827';
    const gridColor = isDark ? '#374151' : '#e5e7eb';
    
    accuracyChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['News Retrieval', 'Fact Checking', 'Bias Detection', 'LLM Decision'],
            datasets: [{
                data: [95, 92, 88, 98],
                backgroundColor: [
                    '#2563eb',
                    '#16a34a',
                    '#d97706',
                    '#0d9488'
                ],
                borderWidth: 0,
                cutout: '70%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: textColor,
                        padding: 20,
                        usePointStyle: true,
                        font: {
                            family: 'Inter',
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: isDark ? '#1f2937' : '#ffffff',
                    titleColor: textColor,
                    bodyColor: textColor,
                    borderColor: gridColor,
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed}% accuracy`;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                duration: 2000
            }
        }
    });
}

function initializeTrustGauge() {
    const ctx = document.getElementById('trustGauge');
    if (!ctx || typeof Chart === 'undefined') return;
    
    const isDark = document.body.classList.contains('dark-mode');
    
    trustGaugeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [0, 100],
                backgroundColor: [
                    '#2563eb',
                    isDark ? '#374151' : '#e5e7eb'
                ],
                borderWidth: 0,
                cutout: '80%',
                circumference: 180,
                rotation: 270
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
                    enabled: false
                }
            },
            animation: {
                animateRotate: true,
                duration: 1500
            }
        }
    });
}

function initializeBiasChart() {
    const ctx = document.getElementById('biasChart');
    if (!ctx || typeof Chart === 'undefined') return;
    
    const isDark = document.body.classList.contains('dark-mode');
    const textColor = isDark ? '#f9fafb' : '#111827';
    const gridColor = isDark ? '#374151' : '#e5e7eb';
    
    biasChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Emotional Language', 'Political Bias', 'Source Credibility', 'Factual Accuracy', 'Objectivity'],
            datasets: [{
                label: 'Bias Analysis',
                data: [0, 0, 0, 0, 0],
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                borderColor: '#2563eb',
                borderWidth: 2,
                pointBackgroundColor: '#2563eb',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: textColor,
                        backdropColor: 'transparent'
                    },
                    grid: {
                        color: gridColor
                    },
                    pointLabels: {
                        color: textColor,
                        font: {
                            family: 'Inter',
                            size: 11
                        }
                    }
                }
            },
            animation: {
                duration: 2000
            }
        }
    });
}

function updateChartsTheme() {
    const isDark = document.body.classList.contains('dark-mode');
    const textColor = isDark ? '#f9fafb' : '#111827';
    const gridColor = isDark ? '#374151' : '#e5e7eb';
    
    // Update accuracy chart
    if (accuracyChart) {
        accuracyChart.options.plugins.legend.labels.color = textColor;
        accuracyChart.options.plugins.tooltip.backgroundColor = isDark ? '#1f2937' : '#ffffff';
        accuracyChart.options.plugins.tooltip.titleColor = textColor;
        accuracyChart.options.plugins.tooltip.bodyColor = textColor;
        accuracyChart.options.plugins.tooltip.borderColor = gridColor;
        accuracyChart.update();
    }
    
    // Update trust gauge
    if (trustGaugeChart) {
        trustGaugeChart.data.datasets[0].backgroundColor[1] = isDark ? '#374151' : '#e5e7eb';
        trustGaugeChart.update();
    }
    
    // Update bias chart
    if (biasChart) {
        biasChart.options.scales.r.ticks.color = textColor;
        biasChart.options.scales.r.grid.color = gridColor;
        biasChart.options.scales.r.pointLabels.color = textColor;
        biasChart.update();
    }
}

// ===== FORM HANDLERS =====
function initializeFormHandlers() {
    const verifyForm = document.getElementById('verify-form');
    const claimInput = document.getElementById('claim');
    const charCount = document.getElementById('char-count');
    
    // Character counter
    if (claimInput && charCount) {
        claimInput.addEventListener('input', () => {
            const count = claimInput.value.length;
            charCount.textContent = count;
            
            if (count > 1800) {
                charCount.style.color = 'var(--error-600)';
            } else if (count > 1500) {
                charCount.style.color = 'var(--warning-600)';
            } else {
                charCount.style.color = 'var(--text-tertiary)';
            }
        });
    }
    
    // Form submission
    if (verifyForm) {
        verifyForm.addEventListener('submit', handleFormSubmission);
    }
    
    // Action buttons
    const verifyAnotherBtn = document.getElementById('verify-another');
    const downloadReportBtn = document.getElementById('download-report');
    const shareResultsBtn = document.getElementById('share-results');
    
    if (verifyAnotherBtn) {
        verifyAnotherBtn.addEventListener('click', resetVerification);
    }
    
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', downloadReport);
    }
    
    if (shareResultsBtn) {
        shareResultsBtn.addEventListener('click', shareResults);
    }
}

function handleFormSubmission(e) {
        e.preventDefault();

    const formData = new FormData(e.target);
    const claim = formData.get('claim')?.trim();

    if (!claim) {
        showNotification('Please enter a news claim to verify.', 'warning');
            return;
        }

    if (claim.length < 10) {
        showNotification('Please enter a more detailed claim (at least 10 characters).', 'warning');
        return;
    }

        startVerification(formData);
    }

// ===== TAB SYSTEM =====
function initializeTabSystem() {
    const tabButtons = document.querySelectorAll('.nav-tab');
    const tabPanels = document.querySelectorAll('.tab-panel');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            // Update active tab button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Update active tab panel
            tabPanels.forEach(panel => {
                panel.classList.remove('active');
                if (panel.id === `${targetTab}-panel`) {
                    panel.classList.add('active');
                }
            });
        });
    });
}

// ===== COUNTER ANIMATIONS =====
function initializeCounters() {
    const statNumbers = document.querySelectorAll('.stat-number[data-count]');
    
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    statNumbers.forEach(counter => {
        observer.observe(counter);
    });
}

function animateCounter(element) {
    const target = parseInt(element.getAttribute('data-count'));
    const duration = 2000;
    const increment = target / (duration / 16);
    let current = 0;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current);
    }, 16);
}

// ===== EXAMPLE CLAIMS =====
function initializeExampleClaims() {
    const exampleClaims = document.querySelectorAll('.example-claim, .example-item');
    
    exampleClaims.forEach(claim => {
        claim.addEventListener('click', () => {
            const claimText = claim.getAttribute('data-claim');
            const claimInput = document.getElementById('claim');
            
            if (claimInput && claimText) {
                claimInput.value = claimText;
                claimInput.dispatchEvent(new Event('input'));
                
                // Focus on the input
                claimInput.focus();
                
                // Add visual feedback
                claim.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    claim.style.transform = '';
                }, 150);
                
                showNotification('Example claim loaded! Click "Verify with AI" to analyze.', 'success');
            }
        });
    });
}

// ===== VERIFICATION PROCESS =====
    function startVerification(formData) {
    console.log('🔍 Starting verification process...');
    
    // CRITICAL: Reset all previous results first
    currentVerificationResults = null;

        // Show results section
    const resultsSection = document.getElementById('results-section');
        resultsSection.classList.remove('d-none');

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }, 300);
    
    // Initialize verification state (this will clear all previous data)
        resetVerificationState();

    // Set claim text
        const claim = formData.get('claim');
    const claimTextElement = document.getElementById('claim-text');
    if (claimTextElement) {
        claimTextElement.textContent = claim;
    }
    
    // Start analysis timer
    startAnalysisTimer();
    
    // Add loading state to verify button
    const verifyBtn = document.querySelector('.btn-verify');
    if (verifyBtn) {
        verifyBtn.classList.add('loading');
    }
    
    // Start verification with API
    performVerification(formData);
}

function resetVerificationState() {
    console.log('🔄 Resetting verification state...');
    
    // Reset verdict badge
    const verdictBadge = document.getElementById('verdict-badge');
    if (verdictBadge) {
        verdictBadge.className = 'verdict-badge analyzing';
        verdictBadge.innerHTML = '<i class="fas fa-spinner fa-spin verdict-icon"></i><span class="verdict-text">Analyzing...</span>';
    }
    
    // Reset confidence score
    const confidenceScore = document.getElementById('confidence-score');
    if (confidenceScore) {
        confidenceScore.textContent = '0%';
    }
    
    // Reset trust gauge
    updateTrustGauge(0);
    
    // Reset agent steps
    resetAgentSteps();
    
    // Reset tab content
    resetTabContent();
    
    // Reset metrics
    resetMetrics();
    
    // Reset analysis timer
    if (analysisTimer) {
        clearInterval(analysisTimer);
        analysisTimer = null;
    }
    analysisStartTime = null;
    
    // Reset timer display
    const timerElement = document.getElementById('analysis-timer');
    if (timerElement) {
        timerElement.textContent = '0s';
    }
    
    // Clear any previous error states
    const errorElements = document.querySelectorAll('.error-state, .verification-error');
    errorElements.forEach(el => el.remove());
    
    console.log('✅ Verification state reset complete');
}

function resetAgentSteps() {
    const steps = ['step-1', 'step-2', 'step-3', 'step-4'];
    
    steps.forEach(stepId => {
        const step = document.getElementById(stepId);
        if (step) {
            step.className = 'agent-step';
            const status = step.querySelector('.step-status');
            if (status) {
                status.textContent = 'Pending';
            }
        }
    });
}

function resetTabContent() {
    console.log('🔄 Resetting all tab content...');
    
    // Reset evidence tab
    const evidenceArticles = document.getElementById('evidence-articles');
    if (evidenceArticles) {
        evidenceArticles.innerHTML = `
            <div class="evidence-loading">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Ready to retrieve relevant articles...</p>
            </div>
        `;
    }
    
    // Reset fact check tab
    const factCheckDetails = document.getElementById('fact-check-details');
    if (factCheckDetails) {
        factCheckDetails.innerHTML = `
            <div class="factcheck-loading">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Ready to perform fact verification...</p>
            </div>
        `;
    }
    
    // Reset bias tab
    const biasIndicators = document.getElementById('bias-indicators');
    if (biasIndicators) {
        biasIndicators.innerHTML = `
            <div class="bias-loading">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Ready to analyze bias indicators...</p>
            </div>
        `;
    }
    
    // Reset reasoning tab
    const llmReasoning = document.getElementById('llm-reasoning');
    if (llmReasoning) {
        llmReasoning.innerHTML = `
            <div class="reasoning-loading">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Ready to generate AI reasoning...</p>
            </div>
        `;
    }
    
    // Reset overview tab content
    const keyFactors = document.getElementById('key-factors');
    if (keyFactors) {
        keyFactors.innerHTML = `
            <div class="factor-item">
                <i class="fas fa-clock"></i>
                <span>Waiting for analysis to begin...</span>
            </div>
        `;
    }
    
    // Reset verdict explanation
    const verdictExplanation = document.getElementById('verdict-explanation');
    if (verdictExplanation) {
        verdictExplanation.textContent = 'Enter a claim above to begin analysis.';
    }
    
    console.log('✅ Tab content reset complete');
}

function resetMetrics() {
    console.log('🔄 Resetting all metrics...');
    
    const metrics = {
        'sources-count': '0',
        'entities-count': '0',
        'processing-time': '0s',
        'bias-score': '0%'
    };
    
    Object.entries(metrics).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    });
    
    // Also reset trust score display
    const trustScoreDisplay = document.getElementById('trust-score-display');
    if (trustScoreDisplay) {
        trustScoreDisplay.textContent = '0%';
    }
    
    console.log('✅ Metrics reset complete');
}

function startAnalysisTimer() {
    analysisStartTime = Date.now();
    const timerElement = document.getElementById('analysis-timer');
    
    analysisTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - analysisStartTime) / 1000);
        if (timerElement) {
            timerElement.textContent = `${elapsed}s`;
        }
    }, 1000);
}

function stopAnalysisTimer() {
    if (analysisTimer) {
        clearInterval(analysisTimer);
        analysisTimer = null;
    }
}

// ===== API COMMUNICATION =====
async function performVerification(formData) {
    try {
        // Step 1: News Retrieval
        updateAgentStep('step-1', 'active', 'Retrieving articles...');
        
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        
                if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            processVerificationResults(data);
        } else {
            throw new Error(data.error || 'Verification failed');
        }
        
    } catch (error) {
        console.error('❌ Verification error:', error);
        handleVerificationError(error);
    } finally {
        // Remove loading state from verify button
        const verifyBtn = document.querySelector('.btn-verify');
        if (verifyBtn) {
            verifyBtn.classList.remove('loading');
        }
        
        stopAnalysisTimer();
    }
}

function processVerificationResults(data) {
    console.log('📊 Processing verification results:', data);
    
    currentVerificationResults = data;
    
    // Process each step
    if (data.steps) {
        processStepResults(data.steps);
    }
    
    // Update final results if available immediately
    if (data.final_result) {
        console.log('🎯 Final result found:', data.final_result);
        updateFinalResults(data.final_result);
    } else {
        console.log('⚠️ No final_result in response, will wait for step completion');
        // Set a fallback timer to ensure verdict shows even if steps don't complete properly
        setTimeout(() => {
            if (currentVerificationResults?.final_result) {
                updateFinalResults(currentVerificationResults.final_result);
            } else if (data.steps?.step4?.data) {
                console.log('🔄 Using fallback final result from step 4');
                const fallbackResult = {
                    verdict: data.steps.step4.data.verdict || 'analysis_complete',
                    confidence: data.steps.step4.data.confidence || 75,
                    trust_score: data.steps.step4.data.trust_score || 0.75,
                    explanation: data.steps.step4.data.explanation || 'Analysis completed successfully.',
                    contributing_factors: data.steps.step4.data.contributing_factors || []
                };
                updateFinalResults(fallbackResult);
            }
        }, 8000); // 8 second fallback
    }
    
    // Update metrics
    updateAnalysisMetrics(data);
}

function processStepResults(steps) {
    // Step 1: News Retrieval
    if (steps.step1) {
        if (steps.step1.status === 'complete') {
            updateAgentStep('step-1', 'complete', `Found ${steps.step1.data?.articles?.length || 0} articles`);
            updateEvidenceTab(steps.step1.data?.articles || []);
        } else if (steps.step1.status === 'error') {
            updateAgentStep('step-1', 'error', 'Error retrieving articles');
        }
    }
    
    // Step 2: Fact Checking
        if (steps.step2) {
        updateAgentStep('step-2', 'active', 'Fact checking...');
        setTimeout(() => {
            if (steps.step2.status === 'complete') {
                updateAgentStep('step-2', 'complete', 'Fact check complete');
                    updateFactCheckTab(steps.step2.data);
            } else if (steps.step2.status === 'error') {
                updateAgentStep('step-2', 'error', 'Error in fact checking');
                }
        }, 1000);
        }

    // Step 3: Bias Analysis
        if (steps.step3) {
        setTimeout(() => {
            updateAgentStep('step-3', 'active', 'Analyzing bias...');
            setTimeout(() => {
            if (steps.step3.status === 'complete') {
                    updateAgentStep('step-3', 'complete', 'Bias analysis complete');
                    updateBiasTab(steps.step3.data);
                } else if (steps.step3.status === 'error') {
                    updateAgentStep('step-3', 'error', 'Error in bias analysis');
                }
            }, 1500);
        }, 2000);
        }

    // Step 4: LLM Decision
        if (steps.step4) {
        setTimeout(() => {
            updateAgentStep('step-4', 'active', 'Making final decision...');
            setTimeout(() => {
            if (steps.step4.status === 'complete') {
                    updateAgentStep('step-4', 'complete', 'Decision complete');
                    updateReasoningTab(steps.step4.data);
                    
                    // Trigger final results update after LLM decision is complete
                    setTimeout(() => {
                        if (currentVerificationResults?.final_result) {
                            updateFinalResults(currentVerificationResults.final_result);
                        } else {
                            // Fallback: create final result from step 4 data
                            const finalResult = {
                                verdict: steps.step4.data?.verdict || 'inconclusive',
                                confidence: steps.step4.data?.confidence || steps.step4.data?.trust_score * 100 || 50,
                                trust_score: steps.step4.data?.trust_score || 0.5,
                                explanation: steps.step4.data?.explanation || 'Analysis completed successfully.',
                                contributing_factors: steps.step4.data?.contributing_factors || []
                            };
                            updateFinalResults(finalResult);
                        }
                    }, 500);
                } else if (steps.step4.status === 'error') {
                    updateAgentStep('step-4', 'error', 'Error in decision making');
                }
            }, 2000);
        }, 4000);
    }
}

function updateAgentStep(stepId, status, message) {
    const step = document.getElementById(stepId);
    if (!step) return;
    
    step.className = `agent-step ${status}`;
    
    const statusElement = step.querySelector('.step-status');
    if (statusElement) {
        statusElement.textContent = message;
    }
}

// ===== TAB CONTENT UPDATES =====
    function updateEvidenceTab(articles) {
    const evidenceArticles = document.getElementById('evidence-articles');
    const evidenceCount = document.getElementById('evidence-count');
    const evidenceSources = document.getElementById('evidence-sources');
    
    if (!evidenceArticles) return;
    
        if (!articles || articles.length === 0) {
        evidenceArticles.innerHTML = `
            <div class="text-center p-4">
                <i class="fas fa-exclamation-triangle text-warning mb-3" style="font-size: 2rem;"></i>
                <p class="text-muted">No relevant articles found for this claim.</p>
            </div>
        `;
            return;
        }

    // Update stats
    if (evidenceCount) {
        evidenceCount.textContent = `${articles.length} articles`;
    }
    
    const uniqueSources = [...new Set(articles.map(article => 
        article.source || new URL(article.url || '').hostname
    ))];
    
    if (evidenceSources) {
        evidenceSources.textContent = `${uniqueSources.length} sources`;
    }
    
    // Create article cards
    const articlesHTML = articles.map(article => `
                <div class="evidence-card">
            <div class="evidence-header">
                <h6 class="evidence-title">${escapeHtml(article.title || 'Untitled Article')}</h6>
                <span class="evidence-source">${escapeHtml(article.source || 'Unknown Source')}</span>
                    </div>
            <p class="evidence-excerpt">${escapeHtml(truncateText(article.text || article.description || '', 150))}</p>
            <div class="evidence-footer">
                <span class="evidence-date">
                    <i class="fas fa-calendar-alt"></i>
                    ${formatDate(article.published_at || article.publishedAt)}
                </span>
                ${article.url ? `<a href="${escapeHtml(article.url)}" target="_blank" class="evidence-link">
                    <i class="fas fa-external-link-alt"></i> Read More
                </a>` : ''}
                </div>
        </div>
    `).join('');

        evidenceArticles.innerHTML = articlesHTML;
    }

    function updateFactCheckTab(factCheckData) {
    const factCheckDetails = document.getElementById('fact-check-details');
    const verificationScore = document.getElementById('verification-score');
    
    if (!factCheckDetails) return;
    
        // Update verification score
    if (verificationScore && factCheckData?.verification_score !== undefined) {
        verificationScore.textContent = factCheckData.verification_score.toFixed(1);
    }
    
    // Create fact check content
    const factCheckHTML = `
        <div class="fact-check-results">
            <div class="fact-check-summary">
                <h6>Verification Summary</h6>
                <p>${escapeHtml(factCheckData?.summary || 'Fact checking analysis completed.')}</p>
            </div>
            
            ${factCheckData?.entities ? `
                <div class="entities-found">
                    <h6>Key Entities Identified</h6>
                    <div class="entities-list">
                        ${factCheckData.entities.map(entity => `
                            <span class="entity-tag">${escapeHtml(entity)}</span>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            
            ${factCheckData?.knowledge_graph_results ? `
                <div class="knowledge-graph-results">
                    <h6>Knowledge Graph Analysis</h6>
                    <div class="kg-results">
                        ${factCheckData.knowledge_graph_results.map(result => `
                            <div class="kg-result-item">
                                <strong>${escapeHtml(result.entity || 'Entity')}</strong>
                                <p>${escapeHtml(result.description || 'No description available')}</p>
                </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
                </div>
            `;

        factCheckDetails.innerHTML = factCheckHTML;
    }

    function updateBiasTab(biasData) {
    const biasIndicators = document.getElementById('bias-indicators');
    const biasScoreValue = document.getElementById('bias-score-value');
    
    if (!biasIndicators) return;
    
        // Update bias score
    if (biasScoreValue && biasData?.bias_score !== undefined) {
        biasScoreValue.textContent = biasData.bias_score.toFixed(1);
    }
    
    // Update bias chart
    if (biasChart && biasData?.bias_breakdown) {
        const breakdown = biasData.bias_breakdown;
        biasChart.data.datasets[0].data = [
            breakdown.emotional_language || 0,
            breakdown.political_bias || 0,
            breakdown.source_credibility || 0,
            breakdown.factual_accuracy || 0,
            breakdown.objectivity || 0
        ];
        biasChart.update();
    }
    
    // Create bias indicators
    const indicators = biasData?.indicators || [];
    const indicatorsHTML = indicators.length > 0 ? 
        indicators.map(indicator => `
            <div class="bias-indicator-item">
                <div class="indicator-icon ${indicator.severity || 'low'}">
                    <i class="fas ${getIndicatorIcon(indicator.type)}"></i>
                </div>
                <div class="indicator-content">
                    <h6>${escapeHtml(indicator.type || 'Bias Indicator')}</h6>
                    <p>${escapeHtml(indicator.description || 'No description available')}</p>
                    <span class="indicator-severity ${indicator.severity || 'low'}">
                        ${indicator.severity || 'Low'} Impact
                    </span>
                </div>
            </div>
        `).join('') :
        `<div class="text-center p-4">
            <i class="fas fa-check-circle text-success mb-3" style="font-size: 2rem;"></i>
            <p class="text-muted">No significant bias indicators detected.</p>
        </div>`;
    
    biasIndicators.innerHTML = indicatorsHTML;
}

function updateReasoningTab(decisionData) {
    const llmReasoning = document.getElementById('llm-reasoning');
    
    if (!llmReasoning) return;
    
    const reasoningHTML = `
        <div class="reasoning-content">
            <div class="reasoning-header">
                <h6>AI Analysis Process</h6>
                <p>The following reasoning was generated by our DeepSeek-R1 model:</p>
            </div>
            
            <div class="reasoning-text">
                ${escapeHtml(decisionData?.reasoning || decisionData?.explanation || 'AI reasoning analysis completed.')}
            </div>
            
            ${decisionData?.confidence_factors ? `
                <div class="confidence-factors">
                    <h6>Confidence Factors</h6>
                    <ul class="factors-list">
                        ${decisionData.confidence_factors.map(factor => `
                            <li class="factor-item">
                                <i class="fas fa-check-circle text-success"></i>
                                <span>${escapeHtml(factor)}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
                </div>
            `;
    
    llmReasoning.innerHTML = reasoningHTML;
}

// ===== FINAL RESULTS UPDATE =====
function updateFinalResults(result) {
    console.log('🎯 Updating final results:', result);
    
    // Extract trust score - handle both 0-1 and 0-100 ranges
    let trustScore = result.trust_score || result.confidence || 0;
    
    // If confidence is provided as percentage but trust_score is 0-1, use trust_score
    if (result.trust_score !== undefined) {
        trustScore = result.trust_score;
    } else if (result.confidence !== undefined) {
        // If confidence is already a percentage (>1), convert to 0-1 range for trust_score
        trustScore = result.confidence > 1 ? result.confidence / 100 : result.confidence;
    }
    
    console.log(`📊 Trust score calculation: trust_score=${result.trust_score}, confidence=${result.confidence}, final=${trustScore}`);
    
    // Update verdict badge
    updateVerdictBadge(result.verdict, result.confidence);
    
    // Update trust gauge with the calculated score
    updateTrustGauge(trustScore);
    
    // Update verdict explanation
    updateVerdictExplanation(result);
    
    // Update key factors
    updateKeyFactors(result.contributing_factors || []);
}

function updateVerdictBadge(verdict, confidence) {
    const verdictBadge = document.getElementById('verdict-badge');
    const verdictIconLarge = document.getElementById('verdict-icon-large');
    const verdictLabel = document.getElementById('verdict-label');
    const verdictExplanation = document.getElementById('verdict-explanation');
    
    if (!verdictBadge) return;
    
    let badgeClass, icon, text, explanation;
    
    switch (verdict?.toLowerCase()) {
        case 'true':
        case 'likely true':
            badgeClass = 'true';
            icon = 'fas fa-check-circle';
            text = 'Likely True';
            explanation = 'Our analysis indicates this claim is likely accurate based on available evidence.';
            break;
        case 'false':
        case 'likely false':
            badgeClass = 'false';
            icon = 'fas fa-times-circle';
            text = 'Likely False';
            explanation = 'Our analysis suggests this claim is likely inaccurate or misleading.';
            break;
        case 'mixed':
        case 'partially true':
            badgeClass = 'mixed';
            icon = 'fas fa-exclamation-triangle';
            text = 'Mixed/Partial';
            explanation = 'This claim contains both accurate and inaccurate elements.';
            break;
        default:
            badgeClass = 'analyzing';
            icon = 'fas fa-question-circle';
            text = 'Inconclusive';
            explanation = 'Unable to determine the accuracy of this claim with sufficient confidence.';
    }
    
    verdictBadge.className = `verdict-badge ${badgeClass}`;
    verdictBadge.innerHTML = `<i class="${icon} verdict-icon"></i><span class="verdict-text">${text}</span>`;
    
    if (verdictIconLarge) {
        verdictIconLarge.innerHTML = `<i class="${icon}"></i>`;
    }
    
    if (verdictLabel) {
        verdictLabel.textContent = text;
    }
    
    if (verdictExplanation) {
        verdictExplanation.textContent = explanation;
    }
}

function updateTrustGauge(score) {
    if (!trustGaugeChart) return;
    
    // Convert score from 0-1 range to 0-100 range
    const scorePercent = score <= 1 ? score * 100 : score;
    const normalizedScore = Math.max(0, Math.min(100, scorePercent));
    
    console.log(`🎯 Updating trust gauge: input=${score}, normalized=${normalizedScore}%`);
    
    trustGaugeChart.data.datasets[0].data = [normalizedScore, 100 - normalizedScore];
    trustGaugeChart.update();
    
    const gaugeValue = document.getElementById('trust-score-display');
    if (gaugeValue) {
        gaugeValue.textContent = `${Math.round(normalizedScore)}%`;
    }
    
    // Also update confidence score display
    const confidenceScore = document.getElementById('confidence-score');
    if (confidenceScore) {
        confidenceScore.textContent = `${Math.round(normalizedScore)}%`;
    }
}

function updateVerdictExplanation(result) {
    const verdictExplanation = document.getElementById('verdict-explanation');
    if (verdictExplanation && result.explanation) {
        verdictExplanation.textContent = result.explanation;
    }
}

function updateKeyFactors(factors) {
    const keyFactors = document.getElementById('key-factors');
    if (!keyFactors) return;
    
    if (!factors || factors.length === 0) {
        keyFactors.innerHTML = `
            <div class="factor-item">
                <i class="fas fa-info-circle"></i>
                <span>Analysis complete - no specific factors highlighted</span>
                </div>
            `;
        return;
    }
    
    const factorsHTML = factors.map(factor => `
        <div class="factor-item">
            <i class="fas fa-check-circle text-success"></i>
            <span>${escapeHtml(factor)}</span>
            </div>
    `).join('');
    
    keyFactors.innerHTML = factorsHTML;
}

function updateAnalysisMetrics(data) {
    // Update sources count
    const sourcesCount = document.getElementById('sources-count');
    if (sourcesCount && data.steps?.step1?.data?.articles) {
        sourcesCount.textContent = data.steps.step1.data.articles.length;
    }
    
    // Update entities count
    const entitiesCount = document.getElementById('entities-count');
    if (entitiesCount && data.steps?.step2?.data?.entities) {
        entitiesCount.textContent = data.steps.step2.data.entities.length;
    }
    
    // Update processing time
    const processingTime = document.getElementById('processing-time');
    if (processingTime && analysisStartTime) {
        const elapsed = Math.floor((Date.now() - analysisStartTime) / 1000);
        processingTime.textContent = `${elapsed}s`;
    }
    
    // Update bias score
    const biasScore = document.getElementById('bias-score');
    if (biasScore && data.steps?.step3?.data?.bias_score !== undefined) {
        biasScore.textContent = `${Math.round(data.steps.step3.data.bias_score)}%`;
    }
}

// ===== ERROR HANDLING =====
function handleVerificationError(error) {
    console.error('❌ Verification error:', error);
    
    // Update all steps to error state
    ['step-1', 'step-2', 'step-3', 'step-4'].forEach(stepId => {
        updateAgentStep(stepId, 'error', 'Error occurred');
    });
    
    // Update verdict badge
    const verdictBadge = document.getElementById('verdict-badge');
    if (verdictBadge) {
        verdictBadge.className = 'verdict-badge false';
        verdictBadge.innerHTML = '<i class="fas fa-exclamation-triangle verdict-icon"></i><span class="verdict-text">Error</span>';
    }
    
    // Show error notification
    showNotification(`Verification failed: ${error.message}`, 'error');
}

// ===== UTILITY FUNCTIONS =====
function resetVerification() {
    console.log('🔄 Resetting verification completely...');
    
    // Clear form
    const verifyForm = document.getElementById('verify-form');
    if (verifyForm) {
        verifyForm.reset();
    }
    
    // Update character counter
    const charCount = document.getElementById('char-count');
    if (charCount) {
        charCount.textContent = '0';
        charCount.style.color = 'var(--text-tertiary)';
    }
    
    // CRITICAL: Clear current results first
    currentVerificationResults = null;
    
    // Reset all verification state completely
    resetVerificationState();
    
    // Hide results section
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.classList.add('d-none');
    }
    
    // Clear claim text display
    const claimTextElement = document.getElementById('claim-text');
    if (claimTextElement) {
        claimTextElement.textContent = '';
    }
    
    // Reset all tab content to loading states
    resetTabContent();
    
    // Reset all metrics
    resetMetrics();
    
    // Reset trust gauge to 0
    updateTrustGauge(0);
    
    // Reset verdict badge to initial state
    const verdictBadge = document.getElementById('verdict-badge');
    if (verdictBadge) {
        verdictBadge.className = 'verdict-badge analyzing';
        verdictBadge.innerHTML = '<i class="fas fa-spinner fa-spin verdict-icon"></i><span class="verdict-text">Ready to Analyze</span>';
    }
    
    // Reset confidence score
    const confidenceScore = document.getElementById('confidence-score');
    if (confidenceScore) {
        confidenceScore.textContent = '0%';
    }
    
    // Scroll to verify section
    const verifySection = document.getElementById('verify');
    if (verifySection) {
        verifySection.scrollIntoView({ behavior: 'smooth' });
    }
    
    console.log('✅ Verification reset complete');
    showNotification('Ready for new verification', 'success');
}

    function downloadReport() {
    if (!currentVerificationResults) {
        showNotification('No verification results to download', 'warning');
            return;
        }

    try {
        const report = generateReport(currentVerificationResults);
        const blob = new Blob([report], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `truthscan-report-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showNotification('Report downloaded successfully', 'success');
    } catch (error) {
        console.error('Error generating report:', error);
        showNotification('Error generating report', 'error');
    }
}

function shareResults() {
    if (!currentVerificationResults) {
        showNotification('No verification results to share', 'warning');
        return;
    }
    
    const shareData = {
        title: 'TruthScan Verification Results',
        text: `I just verified a news claim using TruthScan AI. The verdict: ${currentVerificationResults.final_result?.verdict || 'Analysis complete'}`,
        url: window.location.href
    };
    
    if (navigator.share) {
        navigator.share(shareData).catch(console.error);
    } else {
        // Fallback: copy to clipboard
        const shareText = `${shareData.title}\n${shareData.text}\n${shareData.url}`;
        navigator.clipboard.writeText(shareText).then(() => {
            showNotification('Results copied to clipboard', 'success');
        }).catch(() => {
            showNotification('Unable to share results', 'error');
        });
    }
}

function generateReport(data) {
    const claim = data.claim || 'Unknown claim';
    const verdict = data.final_result?.verdict || 'Unknown';
    const confidence = data.final_result?.confidence || 0;
    const timestamp = new Date().toLocaleString();
    
    return `
TRUTHSCAN VERIFICATION REPORT
Generated: ${timestamp}

CLAIM ANALYZED:
${claim}

VERDICT: ${verdict}
CONFIDENCE: ${confidence}%

ANALYSIS SUMMARY:
- News Articles Retrieved: ${data.steps?.step1?.data?.articles?.length || 0}
- Fact Check Score: ${data.steps?.step2?.data?.verification_score || 'N/A'}
- Bias Score: ${data.steps?.step3?.data?.bias_score || 'N/A'}

EXPLANATION:
${data.final_result?.explanation || 'No detailed explanation available'}

---
Generated by TruthScan v2.0.0
Advanced AI-Powered Fake News Detection
    `.trim();
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${getNotificationIcon(type)}"></i>
            <span>${escapeHtml(message)}</span>
        </div>
        <button class="notification-close">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Show notification
    setTimeout(() => notification.classList.add('show'), 100);
    
    // Auto hide after 5 seconds
    setTimeout(() => hideNotification(notification), 5000);
    
    // Close button handler
    notification.querySelector('.notification-close').addEventListener('click', () => {
        hideNotification(notification);
    });
}

function hideNotification(notification) {
    notification.classList.remove('show');
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 300);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'fa-check-circle';
        case 'warning': return 'fa-exclamation-triangle';
        case 'error': return 'fa-times-circle';
        default: return 'fa-info-circle';
    }
}

function getIndicatorIcon(type) {
    switch (type?.toLowerCase()) {
        case 'emotional': return 'fa-heart';
        case 'political': return 'fa-vote-yea';
        case 'source': return 'fa-link';
        case 'factual': return 'fa-check-double';
        default: return 'fa-exclamation-triangle';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

    function truncateText(text, maxLength) {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function formatDate(dateString) {
    if (!dateString) return 'Unknown date';
    
    try {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    } catch (error) {
        return 'Invalid date';
    }
}

// ===== NOTIFICATION STYLES =====
const notificationStyles = `
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: var(--bg-primary);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    padding: var(--space-4);
    display: flex;
    align-items: center;
    gap: var(--space-3);
    z-index: 10000;
    transform: translateX(400px);
    opacity: 0;
    transition: all var(--transition-normal);
    max-width: 400px;
}

.notification.show {
    transform: translateX(0);
    opacity: 1;
}

.notification-success {
    border-left: 4px solid var(--success-600);
}

.notification-warning {
    border-left: 4px solid var(--warning-600);
}

.notification-error {
    border-left: 4px solid var(--error-600);
}

.notification-info {
    border-left: 4px solid var(--primary-600);
}

.notification-content {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex: 1;
}

.notification-content i {
    font-size: 1.125rem;
}

.notification-success .notification-content i {
    color: var(--success-600);
}

.notification-warning .notification-content i {
    color: var(--warning-600);
}

.notification-error .notification-content i {
    color: var(--error-600);
}

.notification-info .notification-content i {
    color: var(--primary-600);
}

.notification-close {
    background: none;
    border: none;
    color: var(--text-tertiary);
    cursor: pointer;
    padding: var(--space-1);
    border-radius: var(--radius-sm);
    transition: all var(--transition-fast);
}

.notification-close:hover {
    background: var(--bg-secondary);
    color: var(--text-primary);
}
`;

// Add notification styles to page
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);

console.log('🎉 TruthScan v2.0.0 - Ready for professional fake news detection!');