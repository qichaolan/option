/**
 * AI Explainer Module
 *
 * Provides AI-powered explanations for simulation results.
 * Includes local caching, error handling, and mobile-friendly UI.
 */

// =============================================================================
// Configuration
// =============================================================================

const AI_EXPLAINER_CONFIG = {
    endpoint: '/api/ai-explainer',
    cacheTTLMs: 24 * 60 * 60 * 1000, // 24 hours in milliseconds
    requestTimeoutMs: 35000, // 35 seconds (slightly longer than backend)
    retryAttempts: 1,
    retryDelayMs: 2000,
};

// =============================================================================
// Local Cache
// =============================================================================

class AiExplainerCache {
    constructor() {
        this.cache = new Map();
        this.sessionKey = 'ai_explainer_cache';
        this._loadFromSession();
    }

    /**
     * Generate cache key from request parameters
     */
    _generateKey(pageId, contextType, metadata) {
        const keyData = JSON.stringify({
            pageId,
            contextType,
            metadata,
        });
        // Simple hash for cache key
        let hash = 0;
        for (let i = 0; i < keyData.length; i++) {
            const char = keyData.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return `ai_${Math.abs(hash).toString(16)}`;
    }

    /**
     * Load cache from session storage
     */
    _loadFromSession() {
        try {
            const stored = sessionStorage.getItem(this.sessionKey);
            if (stored) {
                const data = JSON.parse(stored);
                const now = Date.now();
                // Only load non-expired entries
                for (const [key, entry] of Object.entries(data)) {
                    if (entry.expiry > now) {
                        this.cache.set(key, entry);
                    }
                }
            }
        } catch (e) {
            console.warn('Failed to load AI cache from session:', e);
        }
    }

    /**
     * Save cache to session storage
     */
    _saveToSession() {
        try {
            const data = {};
            this.cache.forEach((value, key) => {
                data[key] = value;
            });
            sessionStorage.setItem(this.sessionKey, JSON.stringify(data));
        } catch (e) {
            console.warn('Failed to save AI cache to session:', e);
        }
    }

    /**
     * Get cached response if exists and not expired
     */
    get(pageId, contextType, metadata) {
        const key = this._generateKey(pageId, contextType, metadata);
        const entry = this.cache.get(key);

        if (!entry) {
            return null;
        }

        if (Date.now() > entry.expiry) {
            this.cache.delete(key);
            this._saveToSession();
            return null;
        }

        return entry.response;
    }

    /**
     * Store response in cache
     */
    set(pageId, contextType, metadata, response) {
        const key = this._generateKey(pageId, contextType, metadata);
        this.cache.set(key, {
            response,
            expiry: Date.now() + AI_EXPLAINER_CONFIG.cacheTTLMs,
            cachedAt: new Date().toISOString(),
        });
        this._saveToSession();
    }

    /**
     * Clear all cached responses
     */
    clear() {
        this.cache.clear();
        sessionStorage.removeItem(this.sessionKey);
    }
}

// Global cache instance
const aiExplainerCache = new AiExplainerCache();

// =============================================================================
// API Client
// =============================================================================

/**
 * Call the AI Explainer API
 */
async function callAiExplainerApi(pageId, contextType, metadata) {
    const controller = new AbortController();
    const timeoutId = setTimeout(
        () => controller.abort(),
        AI_EXPLAINER_CONFIG.requestTimeoutMs
    );

    try {
        const response = await fetch(AI_EXPLAINER_CONFIG.endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                pageId,
                contextType,
                timestamp: new Date().toISOString(),
                metadata,
            }),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        clearTimeout(timeoutId);

        if (error.name === 'AbortError') {
            throw new Error('Request timed out. Please try again.');
        }

        throw error;
    }
}

/**
 * Get AI explanation with caching and retry logic
 */
async function getAiExplanation(pageId, contextType, metadata) {
    // Check local cache first
    const cached = aiExplainerCache.get(pageId, contextType, metadata);
    if (cached) {
        console.log('AI Explainer: Using cached response');
        return {
            ...cached,
            fromLocalCache: true,
        };
    }

    // Call API with retry logic
    let lastError = null;
    for (let attempt = 0; attempt <= AI_EXPLAINER_CONFIG.retryAttempts; attempt++) {
        try {
            const response = await callAiExplainerApi(pageId, contextType, metadata);

            // Cache successful response
            if (response.success && response.content) {
                aiExplainerCache.set(pageId, contextType, metadata, response);
            }

            return response;
        } catch (error) {
            lastError = error;
            console.warn(`AI Explainer attempt ${attempt + 1} failed:`, error.message);

            if (attempt < AI_EXPLAINER_CONFIG.retryAttempts) {
                await new Promise(resolve =>
                    setTimeout(resolve, AI_EXPLAINER_CONFIG.retryDelayMs)
                );
            }
        }
    }

    throw lastError;
}

// =============================================================================
// UI Components
// =============================================================================

/**
 * State management for AI Explainer panel
 */
const aiExplainerState = {
    isOpen: false,
    isLoading: false,
    lastContent: null,
    lastCachedAt: null,
};

/**
 * Render the AI Explainer button
 */
function renderAiExplainerButton(containerId, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`AI Explainer button container not found: ${containerId}`);
        return null;
    }

    const buttonHtml = `
        <button class="ai-explainer-btn" id="aiExplainerBtn" aria-label="Show Me AI Insights">
            <span class="ai-explainer-btn-icon">&#129302;</span>
            <span class="ai-explainer-btn-text">Show Me AI Insights</span>
        </button>
    `;

    container.innerHTML = buttonHtml;
    return document.getElementById('aiExplainerBtn');
}

/**
 * Render the AI Explainer panel
 */
function renderAiExplainerPanel(containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`AI Explainer panel container not found: ${containerId}`);
        return null;
    }

    const panelHtml = `
        <div class="ai-explainer-panel" id="aiExplainerPanel" style="display: none;">
            <div class="ai-explainer-header">
                <h4 class="ai-explainer-title">
                    <span class="ai-explainer-title-icon">&#129302;</span>
                    AI Analysis
                </h4>
                <div class="ai-explainer-header-actions">
                    <span class="ai-explainer-cache-status" id="aiExplainerCacheStatus"></span>
                    <button class="ai-explainer-collapse" id="aiExplainerCollapse" aria-label="Collapse">
                        &#8722;
                    </button>
                </div>
            </div>
            <div class="ai-explainer-content" id="aiExplainerContent">
                <!-- Content will be inserted here -->
            </div>
        </div>
    `;

    container.innerHTML = panelHtml;
    return {
        panel: document.getElementById('aiExplainerPanel'),
        content: document.getElementById('aiExplainerContent'),
        collapseBtn: document.getElementById('aiExplainerCollapse'),
        cacheStatus: document.getElementById('aiExplainerCacheStatus'),
    };
}

/**
 * Render loading state
 */
function renderLoadingState(contentElement) {
    contentElement.innerHTML = `
        <div class="ai-explainer-loading">
            <div class="ai-explainer-spinner"></div>
            <p class="ai-explainer-loading-text">Analyzing your simulation...</p>
            <p class="ai-explainer-loading-subtext">This may take a few seconds</p>
        </div>
    `;
}

/**
 * Render error state
 */
function renderErrorState(contentElement, errorMessage, onRetry) {
    contentElement.innerHTML = `
        <div class="ai-explainer-error">
            <span class="ai-explainer-error-icon">&#9888;</span>
            <p class="ai-explainer-error-message">${escapeHtml(errorMessage)}</p>
            <button class="ai-explainer-retry-btn" id="aiExplainerRetryBtn">
                Try Again
            </button>
        </div>
    `;

    if (onRetry) {
        document.getElementById('aiExplainerRetryBtn')?.addEventListener('click', onRetry);
    }
}

/**
 * Render credit spread specific content (trade mechanics, key metrics, strategy analysis)
 */
function renderCreditSpreadContent(content) {
    let html = '';

    // Strategy name header
    if (content.strategy_name) {
        html += `<div class="ai-strategy-name">${escapeHtml(content.strategy_name)}</div>`;
    }

    // Trade mechanics
    if (content.trade_mechanics) {
        const tm = content.trade_mechanics;
        // Handle both breakeven (credit spread) and breakevens (iron condor)
        const breakevenLabel = tm.breakevens ? 'Breakevens' : 'Breakeven';
        const breakevenValue = tm.breakevens || tm.breakeven;
        html += `
            <div class="ai-section">
                <h5 class="ai-section-title">Trade Mechanics</h5>
                <div class="ai-trade-mechanics">
                    <div class="ai-mechanic-item">
                        <span class="ai-mechanic-label">Structure:</span>
                        <span class="ai-mechanic-value">${escapeHtml(tm.structure)}</span>
                    </div>
                    <div class="ai-mechanic-item">
                        <span class="ai-mechanic-label">Credit Received:</span>
                        <span class="ai-mechanic-value positive">${escapeHtml(tm.credit_received)}</span>
                    </div>
                    <div class="ai-mechanic-item">
                        <span class="ai-mechanic-label">Margin Requirement:</span>
                        <span class="ai-mechanic-value">${escapeHtml(tm.margin_requirement)}</span>
                    </div>
                    <div class="ai-mechanic-item">
                        <span class="ai-mechanic-label">${breakevenLabel}:</span>
                        <span class="ai-mechanic-value">${escapeHtml(breakevenValue)}</span>
                    </div>
                </div>
            </div>
        `;
    }

    // Key metrics (max profit, max loss, risk/reward)
    if (content.key_metrics) {
        const km = content.key_metrics;
        html += `
            <div class="ai-section">
                <h5 class="ai-section-title">Key Metrics</h5>
                <div class="ai-key-metrics-grid">
                    <div class="ai-metric-card ai-metric-positive">
                        <div class="ai-metric-label">Max Profit</div>
                        <div class="ai-metric-value positive">${escapeHtml(km.max_profit?.value || '-')}</div>
                        <div class="ai-metric-condition">${escapeHtml(km.max_profit?.condition || '')}</div>
                    </div>
                    <div class="ai-metric-card ai-metric-negative">
                        <div class="ai-metric-label">Max Loss</div>
                        <div class="ai-metric-value negative">${escapeHtml(km.max_loss?.value || '-')}</div>
                        <div class="ai-metric-condition">${escapeHtml(km.max_loss?.condition || '')}</div>
                    </div>
                    <div class="ai-metric-card">
                        <div class="ai-metric-label">Risk/Reward</div>
                        <div class="ai-metric-value">${escapeHtml(km.risk_reward_ratio || '-')}</div>
                    </div>
                    <div class="ai-metric-card">
                        <div class="ai-metric-label">Prob. of Profit</div>
                        <div class="ai-metric-value">${escapeHtml(km.probability_of_profit || '-')}</div>
                    </div>
                </div>
            </div>
        `;
    }

    // Visualization (profit zone, loss zone(s), transition zone(s))
    if (content.visualization) {
        const viz = content.visualization;
        const isIronCondor = viz.lower_loss_zone || viz.upper_loss_zone;

        if (isIronCondor) {
            // Iron Condor: two loss zones (lower and upper) and transition zones
            html += `
                <div class="ai-section">
                    <h5 class="ai-section-title">Profit/Loss Zones</h5>
                    <div class="ai-visualization">
                        <div class="ai-viz-item ai-viz-loss">
                            <span class="ai-viz-icon">&#9660;</span>
                            <div class="ai-viz-content">
                                <span class="ai-viz-label">Lower Loss Zone:</span>
                                <span class="ai-viz-value">${escapeHtml(viz.lower_loss_zone || '-')}</span>
                            </div>
                        </div>
                        <div class="ai-viz-item ai-viz-transition">
                            <span class="ai-viz-icon">&#8596;</span>
                            <div class="ai-viz-content">
                                <span class="ai-viz-label">Transition Zones:</span>
                                <span class="ai-viz-value">${escapeHtml(viz.transition_zones || '-')}</span>
                            </div>
                        </div>
                        <div class="ai-viz-item ai-viz-profit">
                            <span class="ai-viz-icon">&#9650;</span>
                            <div class="ai-viz-content">
                                <span class="ai-viz-label">Profit Zone:</span>
                                <span class="ai-viz-value">${escapeHtml(viz.profit_zone || '-')}</span>
                            </div>
                        </div>
                        <div class="ai-viz-item ai-viz-loss">
                            <span class="ai-viz-icon">&#9660;</span>
                            <div class="ai-viz-content">
                                <span class="ai-viz-label">Upper Loss Zone:</span>
                                <span class="ai-viz-value">${escapeHtml(viz.upper_loss_zone || '-')}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            // Credit Spread: single loss zone and transition zone
            html += `
                <div class="ai-section">
                    <h5 class="ai-section-title">Profit/Loss Zones</h5>
                    <div class="ai-visualization">
                        <div class="ai-viz-item ai-viz-profit">
                            <span class="ai-viz-icon">&#9650;</span>
                            <div class="ai-viz-content">
                                <span class="ai-viz-label">Profit Zone:</span>
                                <span class="ai-viz-value">${escapeHtml(viz.profit_zone || '-')}</span>
                            </div>
                        </div>
                        <div class="ai-viz-item ai-viz-transition">
                            <span class="ai-viz-icon">&#8596;</span>
                            <div class="ai-viz-content">
                                <span class="ai-viz-label">Transition Zone:</span>
                                <span class="ai-viz-value">${escapeHtml(viz.transition_zone || '-')}</span>
                            </div>
                        </div>
                        <div class="ai-viz-item ai-viz-loss">
                            <span class="ai-viz-icon">&#9660;</span>
                            <div class="ai-viz-content">
                                <span class="ai-viz-label">Loss Zone:</span>
                                <span class="ai-viz-value">${escapeHtml(viz.loss_zone || '-')}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    // Strategy analysis (bullish, neutral, bearish, and extreme move outcomes)
    if (content.strategy_analysis) {
        const sa = content.strategy_analysis;
        const renderOutcome = (outcome, label, icon) => {
            if (!outcome) return '';
            const sentimentClass = `sentiment-${outcome.sentiment || 'neutral'}`;
            return `
                <div class="ai-outcome ${sentimentClass}">
                    <div class="ai-outcome-header">
                        <span class="ai-outcome-icon">${icon}</span>
                        <strong>${escapeHtml(label)}</strong>
                    </div>
                    <div class="ai-outcome-body">
                        <div class="ai-outcome-scenario">${escapeHtml(outcome.scenario)}</div>
                        <div class="ai-outcome-result">${escapeHtml(outcome.result)}</div>
                    </div>
                </div>
            `;
        };

        html += `
            <div class="ai-section">
                <h5 class="ai-section-title">Strategy Analysis</h5>
                <div class="ai-strategy-analysis">
                    ${renderOutcome(sa.bullish_outcome, 'Bullish Outcome', '&#128200;')}
                    ${renderOutcome(sa.neutral_outcome, 'Neutral Outcome', '&#8596;')}
                    ${renderOutcome(sa.bearish_outcome, 'Bearish Outcome', '&#128201;')}
                    ${renderOutcome(sa.extreme_move_outcome, 'Extreme Move', '&#128165;')}
                </div>
            </div>
        `;
    }

    // Risk management
    if (content.risk_management) {
        const rm = content.risk_management;
        html += `
            <div class="ai-section">
                <h5 class="ai-section-title">Risk Management</h5>
                <div class="ai-risk-management">
                    <div class="ai-rm-item">
                        <span class="ai-rm-icon">&#9888;</span>
                        <div class="ai-rm-content">
                            <span class="ai-rm-label">Early Exit Trigger:</span>
                            <span class="ai-rm-text">${escapeHtml(rm.early_exit_trigger)}</span>
                        </div>
                    </div>
                    <div class="ai-rm-item">
                        <span class="ai-rm-icon">&#128260;</span>
                        <div class="ai-rm-content">
                            <span class="ai-rm-label">Adjustment Options:</span>
                            <span class="ai-rm-text">${escapeHtml(rm.adjustment_options)}</span>
                        </div>
                    </div>
                    <div class="ai-rm-item ai-rm-worst">
                        <span class="ai-rm-icon">&#128683;</span>
                        <div class="ai-rm-content">
                            <span class="ai-rm-label">Worst Case:</span>
                            <span class="ai-rm-text">${escapeHtml(rm.worst_case)}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    return html;
}

/**
 * Check if content is credit spread or iron condor specific
 */
function isCreditSpreadContent(content) {
    return content.trade_mechanics || content.strategy_analysis || content.key_metrics;
}

/**
 * Check if content is specifically an Iron Condor (has two loss zones)
 */
function isIronCondorContent(content) {
    return content.visualization?.lower_loss_zone || content.visualization?.upper_loss_zone ||
           content.strategy_analysis?.extreme_move_outcome;
}

/**
 * Render the explanation content
 */
function renderExplanationContent(contentElement, content, cachedAt) {
    // Build key insights HTML
    const insightsHtml = (content.key_insights || []).map(insight => {
        const sentimentClass = `sentiment-${insight.sentiment || 'neutral'}`;
        const sentimentIcon = {
            positive: '&#9650;', // Up arrow
            negative: '&#9660;', // Down arrow
            neutral: '&#9679;',  // Circle
        }[insight.sentiment || 'neutral'];

        return `
            <div class="ai-insight ${sentimentClass}">
                <div class="ai-insight-header">
                    <span class="ai-insight-icon">${sentimentIcon}</span>
                    <strong class="ai-insight-title">${escapeHtml(insight.title)}</strong>
                </div>
                <p class="ai-insight-description">${escapeHtml(insight.description)}</p>
            </div>
        `;
    }).join('');

    // Build risks HTML
    const risksHtml = (content.risks || []).map(risk => {
        const severityClass = `severity-${risk.severity || 'medium'}`;
        return `
            <div class="ai-risk ${severityClass}">
                <span class="ai-risk-severity">${(risk.severity || 'medium').toUpperCase()}</span>
                <span class="ai-risk-text">${escapeHtml(risk.risk)}</span>
            </div>
        `;
    }).join('');

    // Build watch items HTML
    const watchHtml = (content.watch_items || []).map(item => `
        <div class="ai-watch-item">
            <span class="ai-watch-icon">&#128065;</span>
            <div class="ai-watch-content">
                <span class="ai-watch-text">${escapeHtml(item.item)}</span>
                ${item.trigger ? `<span class="ai-watch-trigger">Trigger: ${escapeHtml(item.trigger)}</span>` : ''}
            </div>
        </div>
    `).join('');

    // Build scenarios HTML (narrative format with historical analysis)
    let scenariosHtml = '';
    if (content.scenarios) {
        const scenarios = content.scenarios;
        const renderScenario = (scenario, scenarioType, icon) => {
            if (!scenario) return '';
            const scenarioLabel = scenarioType === 'medium' ? 'Medium Increase Scenario' : 'Strong Increase Scenario';
            return `
                <div class="ai-scenario">
                    <div class="ai-scenario-header">
                        <span class="ai-scenario-icon">${icon}</span>
                        <strong class="ai-scenario-label">${scenarioLabel} (Min. <span class="ai-scenario-return">${escapeHtml(scenario.min_annual_return)}</span> Annual Return)</strong>
                    </div>
                    <div class="ai-scenario-body">
                        <div class="ai-scenario-paragraph">
                            <span class="ai-scenario-paragraph-label">Projected Price Target:</span>
                            <span class="ai-scenario-paragraph-text">${escapeHtml(scenario.projected_price_target)}</span>
                        </div>
                        <div class="ai-scenario-paragraph">
                            <span class="ai-scenario-paragraph-label">Payoff Realism:</span>
                            <span class="ai-scenario-paragraph-text">${escapeHtml(scenario.payoff_realism)}</span>
                        </div>
                        <div class="ai-scenario-paragraph">
                            <span class="ai-scenario-paragraph-label">Option Payoff:</span>
                            <span class="ai-scenario-paragraph-text">${escapeHtml(scenario.option_payoff)}</span>
                        </div>
                    </div>
                </div>
            `;
        };

        const mediumHtml = renderScenario(scenarios.medium_increase, 'medium', '&#128200;');
        const strongHtml = renderScenario(scenarios.strong_increase, 'strong', '&#128640;');

        if (mediumHtml || strongHtml) {
            scenariosHtml = mediumHtml + strongHtml;
        }
    }

    // Format cached time if available
    let cacheNote = '';
    if (cachedAt) {
        const cachedDate = new Date(cachedAt);
        const timeAgo = formatTimeAgo(cachedDate);
        cacheNote = `<span class="ai-explainer-cache-note">Last updated: ${timeAgo}</span>`;
    }

    // Build credit spread specific content if applicable
    const creditSpreadHtml = isCreditSpreadContent(content) ? renderCreditSpreadContent(content) : '';

    contentElement.innerHTML = `
        <div class="ai-explainer-result">
            <!-- Summary -->
            <div class="ai-summary">
                <p>${escapeHtml(content.summary)}</p>
            </div>

            <!-- Credit Spread Specific Content -->
            ${creditSpreadHtml}

            <!-- Key Insights -->
            ${insightsHtml ? `
                <div class="ai-section">
                    <h5 class="ai-section-title">Key Insights</h5>
                    <div class="ai-insights-list">
                        ${insightsHtml}
                    </div>
                </div>
            ` : ''}

            <!-- Historical Scenarios (LEAPS) -->
            ${scenariosHtml ? `
                <div class="ai-section">
                    <h5 class="ai-section-title">Historical Scenario Analysis</h5>
                    <div class="ai-scenarios-list">
                        ${scenariosHtml}
                    </div>
                </div>
            ` : ''}

            <!-- Risks -->
            ${risksHtml ? `
                <div class="ai-section">
                    <h5 class="ai-section-title">Risk Factors</h5>
                    <div class="ai-risks-list">
                        ${risksHtml}
                    </div>
                </div>
            ` : ''}

            <!-- Watch Items -->
            ${watchHtml ? `
                <div class="ai-section">
                    <h5 class="ai-section-title">What to Watch</h5>
                    <div class="ai-watch-list">
                        ${watchHtml}
                    </div>
                </div>
            ` : ''}

            <!-- Disclaimer -->
            <div class="ai-disclaimer">
                <span class="ai-disclaimer-icon">&#9432;</span>
                <p>${escapeHtml(content.disclaimer || 'This analysis is for educational purposes only.')}</p>
            </div>

            ${cacheNote}
        </div>
    `;
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

/**
 * Format time ago string
 */
function formatTimeAgo(date) {
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
}

// =============================================================================
// Main Controller
// =============================================================================

/**
 * AI Explainer Controller
 *
 * Manages the AI Explainer feature for a specific page/context.
 */
class AiExplainerController {
    constructor(options) {
        this.pageId = options.pageId;
        this.contextType = options.contextType;
        this.getMetadata = options.getMetadata; // Function to get current simulation state (optional)
        this.buttonContainerId = options.buttonContainerId;
        this.panelContainerId = options.panelContainerId;

        this.button = null;
        this.panelElements = null;
        this._storedMetadata = null; // Stored metadata for when setMetadata() is used

        this._init();
    }

    /**
     * Set metadata directly (alternative to getMetadata function)
     * @param {object} metadata - The simulation metadata to use for AI analysis
     */
    setMetadata(metadata) {
        this._storedMetadata = metadata;
    }

    /**
     * Clear the AI panel (hides panel and clears stored metadata)
     */
    clearPanel() {
        this._hidePanel();
        this._storedMetadata = null;
    }

    /**
     * Get the current metadata (from function or stored value)
     */
    _getCurrentMetadata() {
        if (typeof this.getMetadata === 'function') {
            return this.getMetadata();
        }
        return this._storedMetadata;
    }

    /**
     * Initialize the controller
     */
    _init() {
        // Render button
        this.button = renderAiExplainerButton(this.buttonContainerId);
        if (this.button) {
            this.button.addEventListener('click', () => this._handleButtonClick());
        }

        // Render panel
        this.panelElements = renderAiExplainerPanel(this.panelContainerId);
        if (this.panelElements?.collapseBtn) {
            this.panelElements.collapseBtn.addEventListener('click', () => this._togglePanel());
        }

        // Restore panel state from session
        this._restorePanelState();
    }

    /**
     * Handle button click
     */
    async _handleButtonClick() {
        if (aiExplainerState.isLoading) {
            return; // Prevent double-clicks
        }

        const metadata = this._getCurrentMetadata();
        if (!metadata) {
            this._showError('No simulation data available. Please run a simulation first.');
            return;
        }

        await this._fetchExplanation(metadata);
    }

    /**
     * Fetch AI explanation
     */
    async _fetchExplanation(metadata) {
        aiExplainerState.isLoading = true;
        this._updateButtonState(true);
        this._showPanel();
        renderLoadingState(this.panelElements.content);

        try {
            const response = await getAiExplanation(
                this.pageId,
                this.contextType,
                metadata
            );

            if (response.success && response.content) {
                aiExplainerState.lastContent = response.content;
                aiExplainerState.lastCachedAt = response.cachedAt || response.timestamp;

                renderExplanationContent(
                    this.panelElements.content,
                    response.content,
                    response.cached ? response.cachedAt : null
                );

                // Update cache status
                if (response.cached || response.fromLocalCache) {
                    this._updateCacheStatus(response.cachedAt || aiExplainerState.lastCachedAt);
                } else {
                    this._updateCacheStatus(null);
                }
            } else {
                this._showError(response.error || 'Failed to get AI explanation.');
            }
        } catch (error) {
            console.error('AI Explainer error:', error);
            this._showError(error.message || 'An unexpected error occurred.');
        } finally {
            aiExplainerState.isLoading = false;
            this._updateButtonState(false);
        }
    }

    /**
     * Show error in panel
     */
    _showError(message) {
        this._showPanel();
        renderErrorState(
            this.panelElements.content,
            message,
            () => this._handleButtonClick() // Retry handler
        );
    }

    /**
     * Update button loading state
     */
    _updateButtonState(isLoading) {
        if (!this.button) return;

        if (isLoading) {
            this.button.disabled = true;
            this.button.classList.add('loading');
            this.button.querySelector('.ai-explainer-btn-text').textContent = 'Analyzing...';
        } else {
            this.button.disabled = false;
            this.button.classList.remove('loading');
            this.button.querySelector('.ai-explainer-btn-text').textContent = 'Show Me AI Insights';
        }
    }

    /**
     * Update cache status indicator
     */
    _updateCacheStatus(cachedAt) {
        if (!this.panelElements?.cacheStatus) return;

        if (cachedAt) {
            const timeAgo = formatTimeAgo(new Date(cachedAt));
            this.panelElements.cacheStatus.textContent = `Updated ${timeAgo}`;
            this.panelElements.cacheStatus.style.display = 'inline';
        } else {
            this.panelElements.cacheStatus.style.display = 'none';
        }
    }

    /**
     * Show the panel
     */
    _showPanel() {
        if (this.panelElements?.panel) {
            this.panelElements.panel.style.display = 'block';
            aiExplainerState.isOpen = true;
            this._savePanelState();
        }
    }

    /**
     * Hide the panel
     */
    _hidePanel() {
        if (this.panelElements?.panel) {
            this.panelElements.panel.style.display = 'none';
            aiExplainerState.isOpen = false;
            this._savePanelState();
        }
    }

    /**
     * Toggle panel visibility
     */
    _togglePanel() {
        if (aiExplainerState.isOpen) {
            this._hidePanel();
        } else {
            this._showPanel();
        }
    }

    /**
     * Save panel state to session
     */
    _savePanelState() {
        try {
            sessionStorage.setItem(
                `ai_explainer_panel_${this.pageId}`,
                JSON.stringify({ isOpen: aiExplainerState.isOpen })
            );
        } catch (e) {
            // Ignore storage errors
        }
    }

    /**
     * Restore panel state from session
     */
    _restorePanelState() {
        try {
            const stored = sessionStorage.getItem(`ai_explainer_panel_${this.pageId}`);
            if (stored) {
                const state = JSON.parse(stored);
                if (state.isOpen && aiExplainerState.lastContent) {
                    this._showPanel();
                    renderExplanationContent(
                        this.panelElements.content,
                        aiExplainerState.lastContent,
                        aiExplainerState.lastCachedAt
                    );
                }
            }
        } catch (e) {
            // Ignore storage errors
        }
    }

    /**
     * Re-render/update the button state based on current metadata availability
     */
    render() {
        if (!this.button) return;

        const metadata = this._getCurrentMetadata();
        const buttonContainer = document.getElementById(this.buttonContainerId);

        if (metadata) {
            // Show button when simulation data is available
            if (buttonContainer) {
                buttonContainer.style.display = 'block';
            }
            this.button.disabled = false;
        } else {
            // Hide button and panel when no simulation data
            if (buttonContainer) {
                buttonContainer.style.display = 'block'; // Keep container visible but button disabled
            }
            this.button.disabled = true;
            this._hidePanel();
        }
    }

    /**
     * Clear the current explanation (hide panel when new data is selected)
     */
    clearExplanation() {
        this._hidePanel();
    }

    /**
     * Destroy the controller and clean up
     */
    destroy() {
        if (this.button) {
            this.button.remove();
        }
        if (this.panelElements?.panel) {
            this.panelElements.panel.remove();
        }
    }
}

// =============================================================================
// Exports (for use in other modules)
// =============================================================================

// Make available globally
window.AiExplainerController = AiExplainerController;
window.aiExplainerCache = aiExplainerCache;
