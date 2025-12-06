/**
 * Iron Condor Screener - Frontend JavaScript
 *
 * Handles:
 * - Fetching Iron Condor candidates from API
 * - Rendering table and mobile cards
 * - Fetching and displaying payoff charts
 *
 * Units Convention:
 * - Credit: per-share (e.g., $2.50)
 * - Max Profit/Loss: per-contract in dollars (100 shares)
 * - POP/Score: 0-1 internally, displayed as percentage
 * - ROI: 0-1 internally, displayed as percentage
 */

// ============================================
// CONFIGURATION
// ============================================

const API_BASE_URL = '';  // Empty for same-origin requests

// ============================================
// STATE
// ============================================

let state = {
    condors: [],
    loading: false,
    error: null,
    selectedCondorId: null,
    payoffData: null,
    payoffCache: {},  // Cache payoff data by condor ID
    underlyingPrice: 0,
    fetchInProgress: false,  // Prevents double-clicking
    aiExplainerController: null,  // AI Explainer controller instance
    payoffTableExpanded: false,  // Whether P/L table is expanded
};

// ============================================
// DOM ELEMENTS
// ============================================

const elements = {
    // Form inputs
    tickerSelect: document.getElementById('tickerSelect'),
    minRocInput: document.getElementById('minRocInput'),
    minPopInput: document.getElementById('minPopInput'),
    minDte: document.getElementById('minDte'),
    maxDte: document.getElementById('maxDte'),
    runScreenerBtn: document.getElementById('runScreenerBtn'),

    // Info cards
    infoCards: document.getElementById('infoCards'),
    symbolDisplay: document.getElementById('symbolDisplay'),
    underlyingPrice: document.getElementById('underlyingPrice'),
    totalCondors: document.getElementById('totalCondors'),

    // Results
    loadingState: document.getElementById('loadingState'),
    emptyState: document.getElementById('emptyState'),
    condorTableContainer: document.getElementById('condorTableContainer'),
    condorBody: document.getElementById('condorBody'),
    condorMobileCards: document.getElementById('condorMobileCards'),
    noResultsState: document.getElementById('noResultsState'),
    errorDisplay: document.getElementById('errorDisplay'),
    legendSection: document.getElementById('legendSection'),

    // Payoff chart section
    payoffSection: document.getElementById('payoffSection'),
    closePayoff: document.getElementById('closePayoff'),
    payoffSymbol: document.getElementById('payoffSymbol'),
    payoffExpiration: document.getElementById('payoffExpiration'),
    payoffPutSpread: document.getElementById('payoffPutSpread'),
    payoffCallSpread: document.getElementById('payoffCallSpread'),
    payoffCredit: document.getElementById('payoffCredit'),
    payoffRiskReward: document.getElementById('payoffRiskReward'),
    payoffProfitRange: document.getElementById('payoffProfitRange'),
    payoffLegsText: document.getElementById('payoffLegsText'),
    payoffMaxGain: document.getElementById('payoffMaxGain'),
    payoffMaxLoss: document.getElementById('payoffMaxLoss'),
    payoffBreakevens: document.getElementById('payoffBreakevens'),
    payoffTableBody: document.getElementById('payoffTableBody'),
    payoffLoadingState: document.getElementById('payoffLoadingState'),
    togglePayoffTable: document.getElementById('togglePayoffTable'),
};

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    initializeAiExplainer();
    // Auto-fetch on page load
    fetchIronCondors();
});

/**
 * Initialize the AI Explainer controller for Iron Condor Simulator.
 */
function initializeAiExplainer() {
    // Check if AiExplainerController is available
    if (typeof AiExplainerController !== 'undefined') {
        state.aiExplainerController = new AiExplainerController({
            pageId: 'iron_condor_screener',
            contextType: 'spread_simulator',
            buttonContainerId: 'aiExplainerButtonContainer',
            panelContainerId: 'aiExplainerPanelContainer',
        });
    } else {
        console.warn('AiExplainerController not found. AI Explainer disabled.');
    }
}

function setupEventListeners() {
    // Run screener button
    if (elements.runScreenerBtn) {
        elements.runScreenerBtn.addEventListener('click', fetchIronCondors);
    }

    // Close payoff button
    if (elements.closePayoff) {
        elements.closePayoff.addEventListener('click', closePayoffChart);
    }

    // Toggle P/L table expand/collapse
    if (elements.togglePayoffTable) {
        elements.togglePayoffTable.addEventListener('click', togglePayoffTableExpand);
    }

    // Re-render on window resize (debounced)
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            if (state.condors.length > 0) {
                renderResults();
            }
        }, 250);
    });

    // Use event delegation for table rows (avoids re-adding listeners)
    if (elements.condorBody) {
        elements.condorBody.addEventListener('click', handleTableRowClick);
    }
    if (elements.condorMobileCards) {
        elements.condorMobileCards.addEventListener('click', handleMobileCardClick);
    }
}

function handleTableRowClick(event) {
    const row = event.target.closest('tr[data-condor-id]');
    if (row) {
        const condorId = row.dataset.condorId;
        fetchPayoff(condorId);
        highlightRow(condorId);
    }
}

function handleMobileCardClick(event) {
    const card = event.target.closest('.mobile-spread-card[data-condor-id]');
    if (card) {
        const condorId = card.dataset.condorId;
        fetchPayoff(condorId);
        highlightCard(condorId);
    }
}

// ============================================
// API INTERACTION
// ============================================

async function fetchIronCondors() {
    // Prevent double-clicking
    if (state.fetchInProgress) {
        return;
    }

    const symbol = elements.tickerSelect?.value || 'QQQ';
    const minRoc = parseFloat(elements.minRocInput?.value || '0.15');
    const minPop = parseFloat(elements.minPopInput?.value || '0.50');
    const dteMin = parseInt(elements.minDte?.value || '14', 10);
    const dteMax = parseInt(elements.maxDte?.value || '45', 10);

    // Validate inputs
    if (isNaN(minRoc) || isNaN(minPop) || isNaN(dteMin) || isNaN(dteMax)) {
        showError('Invalid filter values. Please check your inputs.');
        return;
    }

    if (minRoc < 0 || minRoc > 1 || minPop < 0 || minPop > 1) {
        showError('ROC and POP must be between 0 and 100%.');
        return;
    }

    state.fetchInProgress = true;
    state.loading = true;
    state.error = null;
    state.payoffCache = {};  // Clear payoff cache on new scan
    hideError();
    showLoading();
    setButtonLoading(true);
    closePayoffChart();

    try {
        const params = new URLSearchParams({
            symbol,
            dte_min: String(dteMin),
            dte_max: String(dteMax),
            min_roc: String(minRoc),
            min_pop: String(minPop),
            limit: '20',
        });

        const response = await fetch(`${API_BASE_URL}/api/iron-condors?${params}`);

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${response.status})`);
        }

        const data = await response.json();

        // Validate response data
        if (!data || typeof data !== 'object') {
            throw new Error('Invalid response from server');
        }

        updateUI(data);

    } catch (err) {
        console.error('Error fetching Iron Condors:', err);
        state.error = err.message;
        showError(err.message);
        hideLoading();
        if (elements.emptyState) {
            elements.emptyState.style.display = 'block';
        }
    } finally {
        state.loading = false;
        state.fetchInProgress = false;
        setButtonLoading(false);
    }
}

async function fetchPayoff(condorId) {
    if (!condorId) return;

    // Clear AI Explainer panel when selecting a new condor
    if (state.aiExplainerController) {
        state.aiExplainerController.clearPanel();
    }

    // Reset table to collapsed state when selecting a new condor
    state.payoffTableExpanded = false;

    state.selectedCondorId = condorId;

    // Check cache first
    if (state.payoffCache[condorId]) {
        state.payoffData = state.payoffCache[condorId];
        updatePayoffUI(state.payoffData);
        if (elements.payoffSection) {
            elements.payoffSection.style.display = 'block';
        }
        if (isMobile() && elements.payoffSection) {
            elements.payoffSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        return;
    }

    if (elements.payoffSection) {
        elements.payoffSection.style.display = 'block';
    }
    if (elements.payoffLoadingState) {
        elements.payoffLoadingState.style.display = 'flex';
    }

    try {
        // Request ±8% range for the payoff curve
        const params = new URLSearchParams({
            move_low_pct: '-0.08',
            move_high_pct: '0.08',
        });
        const response = await fetch(
            `${API_BASE_URL}/api/iron-condors/${encodeURIComponent(condorId)}/payoff?${params}`
        );

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            // Provide user-friendly message for cache miss (404)
            if (response.status === 404) {
                throw new Error('This Iron Condor is no longer available. Please run a new scan.');
            }
            throw new Error(err.detail || `Failed to load payoff (${response.status})`);
        }

        const data = await response.json();

        // Validate response
        if (!data || !Array.isArray(data.points)) {
            throw new Error('Invalid payoff data received');
        }

        state.payoffData = data;
        state.payoffCache[condorId] = data;  // Cache for reuse
        updatePayoffUI(data);

        // Scroll to payoff section on mobile
        if (isMobile() && elements.payoffSection) {
            elements.payoffSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

    } catch (err) {
        console.error('Error fetching payoff:', err);
        showError(err.message);
        closePayoffChart();
    } finally {
        if (elements.payoffLoadingState) {
            elements.payoffLoadingState.style.display = 'none';
        }
    }
}

// ============================================
// UI UPDATE FUNCTIONS
// ============================================

function updateUI(data) {
    state.condors = Array.isArray(data.candidates) ? data.candidates : [];
    state.underlyingPrice = safeNumber(data.underlying_price, 0);

    // Update summary cards
    if (elements.symbolDisplay) {
        elements.symbolDisplay.textContent = escapeHtml(data.symbol || '-');
    }
    if (elements.underlyingPrice) {
        elements.underlyingPrice.textContent = formatCurrency(state.underlyingPrice);
    }
    if (elements.totalCondors) {
        elements.totalCondors.textContent = String(data.total_candidates || 0);
    }

    // Show info cards
    if (elements.infoCards) {
        elements.infoCards.style.display = 'grid';
    }

    hideLoading();

    if (state.condors.length === 0) {
        if (elements.emptyState) elements.emptyState.style.display = 'none';
        if (elements.condorTableContainer) elements.condorTableContainer.style.display = 'none';
        if (elements.noResultsState) elements.noResultsState.style.display = 'block';
        if (elements.legendSection) elements.legendSection.style.display = 'none';
    } else {
        if (elements.emptyState) elements.emptyState.style.display = 'none';
        if (elements.noResultsState) elements.noResultsState.style.display = 'none';
        if (elements.condorTableContainer) elements.condorTableContainer.style.display = 'block';
        if (elements.legendSection) elements.legendSection.style.display = 'block';
        renderResults();

        // Auto-select first condor
        const firstCondor = state.condors[0];
        if (firstCondor && firstCondor.id) {
            fetchPayoff(firstCondor.id);
            highlightRow(firstCondor.id);
        }
    }
}

function renderResults() {
    if (isMobile()) {
        renderMobileCards(state.condors);
        if (elements.condorMobileCards) {
            elements.condorMobileCards.style.display = 'flex';
        }
        if (elements.condorBody && elements.condorBody.parentElement) {
            elements.condorBody.parentElement.style.display = 'none';
        }
    } else {
        renderTable(state.condors);
        if (elements.condorMobileCards) {
            elements.condorMobileCards.style.display = 'none';
        }
        if (elements.condorBody && elements.condorBody.parentElement) {
            elements.condorBody.parentElement.style.display = 'table';
        }
    }
}

function renderTable(condors) {
    if (!elements.condorBody) return;

    const top20 = condors.slice(0, 20);
    const selectedId = state.selectedCondorId;

    elements.condorBody.innerHTML = top20.map((c, idx) => {
        const isTop = idx === 0;
        const isSelected = c.id === selectedId;
        let rowClass = 'clickable-row';
        if (isTop) rowClass += ' top-ranked-row';
        if (isSelected) rowClass += ' selected-for-sim';

        // Safe value extraction with defaults
        const shortPut = safeNumber(c.short_put);
        const longPut = safeNumber(c.long_put);
        const shortCall = safeNumber(c.short_call);
        const longCall = safeNumber(c.long_call);
        const totalCredit = safeNumber(c.total_credit);
        const maxProfit = safeNumber(c.max_profit);
        const maxLoss = safeNumber(c.max_loss);
        const riskReward = safeNumber(c.risk_reward_ratio);
        const pop = safeNumber(c.combined_pop);
        const score = safeNumber(c.combined_score);

        return `
            <tr data-condor-id="${escapeHtml(c.id)}" class="${rowClass}"
                title="Click to see payoff chart" tabindex="0" role="button">
                <td class="col-exp">${escapeHtml(c.expiration || '-')}</td>
                <td class="col-dte">${safeNumber(c.dte, 0)}</td>
                <td class="col-strike">${formatCurrency(shortPut, 0)}</td>
                <td class="col-strike hide-mobile">${formatCurrency(longPut, 0)}</td>
                <td class="col-strike">${formatCurrency(shortCall, 0)}</td>
                <td class="col-strike hide-mobile">${formatCurrency(longCall, 0)}</td>
                <td class="col-money">${formatCurrency(totalCredit, 2)}</td>
                <td class="col-money hide-mobile">
                    <span class="positive">+${formatCurrency(maxProfit, 0)}</span>
                    <span class="text-divider">/</span>
                    <span class="negative">-${formatCurrency(maxLoss, 0)}</span>
                </td>
                <td class="col-pct">${formatPercent(riskReward, 0)}</td>
                <td class="col-pct ${pop >= 0.60 ? 'positive' : ''}">${formatPercent(pop, 0)}</td>
                <td class="col-score">
                    <span class="score-badge ${getScoreClass(score)}">${formatNumber(score, 2)}</span>
                </td>
            </tr>
        `;
    }).join('');
}

function renderMobileCards(condors) {
    if (!elements.condorMobileCards) return;

    const top20 = condors.slice(0, 20);
    const selectedId = state.selectedCondorId;

    elements.condorMobileCards.innerHTML = top20.map((c, idx) => {
        const isTop = idx === 0;
        const isSelected = c.id === selectedId;
        let cardClass = 'mobile-spread-card clickable-card';
        if (isTop) cardClass += ' top-ranked-card';
        if (isSelected) cardClass += ' selected-for-sim';

        // Safe value extraction
        const shortPut = safeNumber(c.short_put);
        const longPut = safeNumber(c.long_put);
        const shortCall = safeNumber(c.short_call);
        const longCall = safeNumber(c.long_call);
        const totalCredit = safeNumber(c.total_credit);
        const maxProfit = safeNumber(c.max_profit);
        const maxLoss = safeNumber(c.max_loss);
        const pop = safeNumber(c.combined_pop);
        const score = safeNumber(c.combined_score);

        return `
            <div class="${cardClass}" data-condor-id="${escapeHtml(c.id)}"
                 tabindex="0" role="button" aria-label="Iron Condor ${idx + 1}">
                <div class="mobile-card-header">
                    <div>
                        <div class="mobile-card-strikes">
                            Put: ${formatCurrency(shortPut, 0)}/${formatCurrency(longPut, 0)}
                            &nbsp;|&nbsp;
                            Call: ${formatCurrency(shortCall, 0)}/${formatCurrency(longCall, 0)}
                        </div>
                        <div class="mobile-card-exp">
                            ${escapeHtml(c.expiration || '-')} (${safeNumber(c.dte, 0)} DTE)
                        </div>
                    </div>
                    <div class="mobile-card-score">
                        <span class="score-badge ${getScoreClass(score)}">
                            ${formatNumber(score, 2)}
                        </span>
                    </div>
                </div>
                <div class="mobile-card-metrics">
                    <div class="mobile-metric">
                        <div class="mobile-metric-label">Credit/sh</div>
                        <div class="mobile-metric-value">${formatCurrency(totalCredit, 2)}</div>
                    </div>
                    <div class="mobile-metric">
                        <div class="mobile-metric-label">Max Profit</div>
                        <div class="mobile-metric-value positive">+${formatCurrency(maxProfit, 0)}</div>
                    </div>
                    <div class="mobile-metric">
                        <div class="mobile-metric-label">Max Loss</div>
                        <div class="mobile-metric-value negative">-${formatCurrency(maxLoss, 0)}</div>
                    </div>
                    <div class="mobile-metric">
                        <div class="mobile-metric-label">POP</div>
                        <div class="mobile-metric-value">${formatPercent(pop, 0)}</div>
                    </div>
                </div>
                <div class="mobile-card-action">
                    <span class="tap-to-simulate">Tap to simulate P/L &rarr;</span>
                </div>
            </div>
        `;
    }).join('');
}

function highlightRow(condorId) {
    // Remove previous selection
    document.querySelectorAll('.selected-for-sim').forEach(el => {
        el.classList.remove('selected-for-sim');
    });

    // Highlight new selection
    if (condorId) {
        const row = document.querySelector(`tr[data-condor-id="${CSS.escape(condorId)}"]`);
        if (row) {
            row.classList.add('selected-for-sim');
        }
        const card = document.querySelector(`.mobile-spread-card[data-condor-id="${CSS.escape(condorId)}"]`);
        if (card) {
            card.classList.add('selected-for-sim');
        }
    }
}

function highlightCard(condorId) {
    highlightRow(condorId);  // Same logic for both
}

function updatePayoffUI(data) {
    if (!data) return;

    // Safe value extraction
    const symbol = data.symbol || '-';
    const expiration = data.expiration || '-';
    const shortPut = safeNumber(data.short_put);
    const longPut = safeNumber(data.long_put);
    const shortCall = safeNumber(data.short_call);
    const longCall = safeNumber(data.long_call);
    const totalCredit = safeNumber(data.total_credit);
    const riskReward = safeNumber(data.risk_reward_ratio);
    const maxProfit = safeNumber(data.max_profit);
    const maxLoss = safeNumber(data.max_loss);
    const breakEvenLow = safeNumber(data.breakeven_low);
    const breakEvenHigh = safeNumber(data.breakeven_high);

    // Update info grid
    if (elements.payoffSymbol) {
        elements.payoffSymbol.textContent = escapeHtml(symbol);
    }
    if (elements.payoffExpiration) {
        elements.payoffExpiration.textContent = escapeHtml(expiration);
    }
    if (elements.payoffPutSpread) {
        elements.payoffPutSpread.textContent =
            `${formatCurrency(shortPut, 0)} / ${formatCurrency(longPut, 0)}`;
    }
    if (elements.payoffCallSpread) {
        elements.payoffCallSpread.textContent =
            `${formatCurrency(shortCall, 0)} / ${formatCurrency(longCall, 0)}`;
    }
    if (elements.payoffCredit) {
        elements.payoffCredit.textContent = `${formatCurrency(totalCredit, 2)}/share`;
    }
    if (elements.payoffRiskReward) {
        elements.payoffRiskReward.textContent = formatPercent(riskReward, 0);
    }

    // Update action summary - compact two-line format
    if (elements.payoffProfitRange) {
        elements.payoffProfitRange.textContent =
            `${formatCurrency(breakEvenLow, 2)} – ${formatCurrency(breakEvenHigh, 2)}`;
    }
    if (elements.payoffLegsText) {
        elements.payoffLegsText.innerHTML =
            `Sell ${formatCurrency(shortPut, 0)} / Buy ${formatCurrency(longPut, 0)} puts` +
            `<span class="action-separator">|</span>` +
            `Sell ${formatCurrency(shortCall, 0)} / Buy ${formatCurrency(longCall, 0)} calls`;
    }

    // Update summary cards (these are per-contract values)
    if (elements.payoffMaxGain) {
        elements.payoffMaxGain.textContent = '+' + formatCurrency(maxProfit, 0);
    }
    if (elements.payoffMaxLoss) {
        elements.payoffMaxLoss.textContent = '-' + formatCurrency(maxLoss, 0);
    }
    if (elements.payoffBreakevens) {
        elements.payoffBreakevens.textContent =
            `${formatCurrency(breakEvenLow, 2)} - ${formatCurrency(breakEvenHigh, 2)}`;
    }

    // Render table
    const points = Array.isArray(data.points) ? data.points : [];
    renderPayoffTable(points);

    // Update AI Explainer with Iron Condor context
    if (state.aiExplainerController) {
        state.aiExplainerController.setMetadata({
            symbol: symbol,
            underlying_price: state.underlyingPrice,
            expiration: expiration,
            short_put_strike: shortPut,
            long_put_strike: longPut,
            short_call_strike: shortCall,
            long_call_strike: longCall,
            net_credit: totalCredit,
            max_profit: maxProfit,
            max_loss: maxLoss,
            breakeven_low: breakEvenLow,
            breakeven_high: breakEvenHigh,
            risk_reward_ratio: riskReward,
            points: points.map(p => ({
                move_pct: p.move_pct,
                price: p.price,
                payoff: p.payoff,
                roi: p.roi,
            })),
        });
    }
}

function renderPayoffTable(points) {
    if (!elements.payoffTableBody) return;

    if (!points || points.length === 0) {
        elements.payoffTableBody.innerHTML =
            '<tr><td colspan="4" class="text-center">No data</td></tr>';
        if (elements.togglePayoffTable) {
            elements.togglePayoffTable.style.display = 'none';
        }
        return;
    }

    // Key moves to show in collapsed state: -5%, 0%, +5%
    const keyMoves = [-0.05, 0, 0.05];

    // Filter points for collapsed view
    let displayPoints = points;
    if (!state.payoffTableExpanded) {
        displayPoints = points.filter(p => {
            const movePct = safeNumber(p.move_pct, 0);
            return keyMoves.some(key => Math.abs(movePct - key) < 0.005);
        });
        // Fallback: if no key points found, show first 3 points
        if (displayPoints.length === 0) {
            displayPoints = points.slice(0, 3);
        }
    }

    elements.payoffTableBody.innerHTML = displayPoints.map(point => {
        const payoff = safeNumber(point.payoff, 0);
        const price = safeNumber(point.price, 0);
        const movePct = safeNumber(point.move_pct, 0) * 100;
        const roi = safeNumber(point.roi, 0) * 100;

        const plClass = payoff >= 0 ? 'positive' : 'negative';
        const plPrefix = payoff >= 0 ? '+' : '';
        const pctPrefix = movePct >= 0 ? '+' : '';
        const roiPrefix = roi >= 0 ? '+' : '';

        return `
            <tr>
                <td class="col-pct-move">${pctPrefix}${movePct.toFixed(0)}%</td>
                <td class="col-price">${formatCurrency(price, 2)}</td>
                <td class="col-pl ${plClass}">${plPrefix}${formatCurrency(payoff, 0)}</td>
                <td class="col-pct ${plClass}">${roiPrefix}${roi.toFixed(1)}%</td>
            </tr>
        `;
    }).join('');

    // Show/hide toggle button and update its text
    if (elements.togglePayoffTable) {
        // Only show toggle if there are more points than displayed
        if (points.length > displayPoints.length || state.payoffTableExpanded) {
            elements.togglePayoffTable.style.display = 'block';
            const icon = elements.togglePayoffTable.querySelector('.toggle-icon');
            const text = elements.togglePayoffTable.querySelector('.toggle-text');
            if (state.payoffTableExpanded) {
                if (icon) icon.innerHTML = '&#9650;';  // Up arrow
                if (text) text.textContent = 'Show less';
            } else {
                if (icon) icon.innerHTML = '&#9660;';  // Down arrow
                if (text) text.textContent = 'Show full P/L table';
            }
        } else {
            elements.togglePayoffTable.style.display = 'none';
        }
    }
}

/**
 * Toggle P/L table between collapsed (3 key rows) and expanded (all rows) state.
 */
function togglePayoffTableExpand() {
    state.payoffTableExpanded = !state.payoffTableExpanded;

    // Re-render the table with current payoff data
    if (state.payoffData && Array.isArray(state.payoffData.points)) {
        renderPayoffTable(state.payoffData.points);
    }
}

function closePayoffChart() {
    if (elements.payoffSection) {
        elements.payoffSection.style.display = 'none';
    }
    state.selectedCondorId = null;
    state.payoffData = null;
    state.payoffTableExpanded = false;  // Reset to collapsed when closing

    // Remove selection highlighting
    document.querySelectorAll('.selected-for-sim').forEach(el => {
        el.classList.remove('selected-for-sim');
    });

    // Clear AI Explainer panel
    if (state.aiExplainerController) {
        state.aiExplainerController.clearPanel();
    }
}

// ============================================
// HELPER FUNCTIONS
// ============================================

/**
 * Safely convert a value to a number, returning default if NaN or null/undefined.
 */
function safeNumber(value, defaultValue = 0) {
    if (value === null || value === undefined) return defaultValue;
    const num = Number(value);
    return isNaN(num) ? defaultValue : num;
}

function getScoreClass(score) {
    const s = safeNumber(score, 0);
    if (s >= 0.6) return 'score-high';
    if (s >= 0.4) return 'score-medium';
    return 'score-low';
}

function formatCurrency(value, decimals = 2) {
    const num = safeNumber(value);
    if (num === 0 && value !== 0 && value !== '0') return '-';
    return '$' + num.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    });
}

function formatPercent(value, decimals = 0) {
    const num = safeNumber(value);
    return (num * 100).toFixed(decimals) + '%';
}

function formatNumber(value, decimals = 2) {
    const num = safeNumber(value);
    return num.toFixed(decimals);
}

function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const str = String(text);
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function isMobile() {
    return window.innerWidth < 768;
}

function showLoading() {
    if (elements.loadingState) elements.loadingState.style.display = 'block';
    if (elements.emptyState) elements.emptyState.style.display = 'none';
    if (elements.condorTableContainer) elements.condorTableContainer.style.display = 'none';
    if (elements.noResultsState) elements.noResultsState.style.display = 'none';
}

function hideLoading() {
    if (elements.loadingState) elements.loadingState.style.display = 'none';
}

function showError(message) {
    if (!elements.errorDisplay) return;

    const errorMsg = elements.errorDisplay.querySelector('.error-message');
    if (errorMsg) {
        errorMsg.textContent = message || 'An error occurred';
    } else {
        elements.errorDisplay.textContent = message || 'An error occurred';
    }
    elements.errorDisplay.style.display = 'flex';
}

function hideError() {
    if (elements.errorDisplay) {
        elements.errorDisplay.style.display = 'none';
    }
}

function setButtonLoading(isLoading) {
    const btn = elements.runScreenerBtn;
    if (!btn) return;

    if (isLoading) {
        btn.disabled = true;
        btn.classList.add('btn-loading');
        btn.innerHTML = '<span class="loading-spinner-sm"></span> Scanning...';
    } else {
        btn.disabled = false;
        btn.classList.remove('btn-loading');
        btn.innerHTML = '<span class="btn-icon">&#9654;</span> Scan Iron Condors';
    }
}
