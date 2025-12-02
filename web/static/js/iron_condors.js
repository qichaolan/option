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
    payoffProfitCondition: document.getElementById('payoffProfitCondition'),
    payoffTradeExplanation: document.getElementById('payoffTradeExplanation'),
    payoffMaxGain: document.getElementById('payoffMaxGain'),
    payoffMaxLoss: document.getElementById('payoffMaxLoss'),
    payoffBreakevens: document.getElementById('payoffBreakevens'),
    payoffChart: document.getElementById('payoffChart'),
    payoffTableBody: document.getElementById('payoffTableBody'),
    payoffLoadingState: document.getElementById('payoffLoadingState'),
};

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    // Auto-fetch on page load
    fetchIronCondors();
});

function setupEventListeners() {
    // Run screener button
    if (elements.runScreenerBtn) {
        elements.runScreenerBtn.addEventListener('click', fetchIronCondors);
    }

    // Close payoff button
    if (elements.closePayoff) {
        elements.closePayoff.addEventListener('click', closePayoffChart);
    }

    // Re-render on window resize (debounced)
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            if (state.condors.length > 0) {
                renderResults();
            }
            if (state.payoffData && elements.payoffSection &&
                elements.payoffSection.style.display !== 'none') {
                renderPayoffChart(state.payoffData.points);
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
        const response = await fetch(
            `${API_BASE_URL}/api/iron-condors/${encodeURIComponent(condorId)}/payoff`
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
        elements.infoCards.style.display = 'flex';
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
                    <span class="tap-to-simulate">Tap to see payoff chart &rarr;</span>
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

    // Update action summary
    if (elements.payoffProfitCondition) {
        elements.payoffProfitCondition.textContent =
            `Profit if ${escapeHtml(symbol)} stays between ` +
            `${formatCurrency(breakEvenLow, 2)} and ${formatCurrency(breakEvenHigh, 2)} at expiration.`;
    }
    if (elements.payoffTradeExplanation) {
        elements.payoffTradeExplanation.textContent =
            `Sell put ${formatCurrency(shortPut, 0)}, buy put ${formatCurrency(longPut, 0)} + ` +
            `Sell call ${formatCurrency(shortCall, 0)}, buy call ${formatCurrency(longCall, 0)}.`;
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

    // Render chart and table
    const points = Array.isArray(data.points) ? data.points : [];
    renderPayoffChart(points);
    renderPayoffTable(points);
}

function renderPayoffChart(points) {
    if (!elements.payoffChart) return;

    if (!points || points.length === 0) {
        elements.payoffChart.innerHTML = '<div class="chart-no-data">No data available</div>';
        return;
    }

    // Find max absolute P/L for scaling (avoid division by zero)
    const payoffs = points.map(p => safeNumber(p.payoff, 0));
    const maxAbsPL = Math.max(...payoffs.map(p => Math.abs(p)), 1);
    const chartHeight = isMobile() ? 120 : 180;
    const halfHeight = chartHeight / 2;

    // Limit number of bars to avoid overcrowding
    const maxBars = isMobile() ? 7 : 11;
    const step = points.length > maxBars ? Math.ceil(points.length / maxBars) : 1;
    const displayPoints = points.filter((_, idx) => idx % step === 0);

    // Generate bar chart HTML
    let barsHtml = '<div class="pl-chart-zero-line"></div>';

    displayPoints.forEach((point) => {
        const payoff = safeNumber(point.payoff, 0);
        const isPositive = payoff >= 0;
        const barHeight = (Math.abs(payoff) / maxAbsPL) * halfHeight * 0.85;

        const barStyle = isPositive
            ? `height: ${barHeight}px; bottom: 50%;`
            : `height: ${barHeight}px; top: 50%;`;

        const valueClass = isPositive ? 'positive' : 'negative';
        const valueStyle = isPositive
            ? `bottom: calc(50% + ${barHeight + 3}px);`
            : `top: calc(50% + ${barHeight + 3}px);`;

        const movePct = safeNumber(point.move_pct, 0) * 100;
        const pctLabel = movePct >= 0 ? `+${movePct.toFixed(0)}%` : `${movePct.toFixed(0)}%`;

        barsHtml += `
            <div class="pl-chart-bar-container">
                <div class="pl-chart-bar ${valueClass}"
                     style="${barStyle}"
                     title="${pctLabel}: ${formatCurrency(payoff, 0)}">
                </div>
                <div class="pl-chart-value ${valueClass}" style="${valueStyle}">
                    ${formatCurrency(payoff, 0)}
                </div>
                <div class="pl-chart-label">${pctLabel}</div>
            </div>
        `;
    });

    elements.payoffChart.innerHTML = `
        <div class="pl-chart-bars" style="height: ${chartHeight}px;">
            ${barsHtml}
        </div>
    `;
}

function renderPayoffTable(points) {
    if (!elements.payoffTableBody) return;

    if (!points || points.length === 0) {
        elements.payoffTableBody.innerHTML =
            '<tr><td colspan="4" class="text-center">No data</td></tr>';
        return;
    }

    elements.payoffTableBody.innerHTML = points.map(point => {
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
}

function closePayoffChart() {
    if (elements.payoffSection) {
        elements.payoffSection.style.display = 'none';
    }
    state.selectedCondorId = null;
    state.payoffData = null;

    // Remove selection highlighting
    document.querySelectorAll('.selected-for-sim').forEach(el => {
        el.classList.remove('selected-for-sim');
    });
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
