/**
 * LEAPS Ranker - Frontend Application
 */

// Utility: Escape HTML to prevent XSS
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

// Utility: Format number safely
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return '-';
    return Number(num).toFixed(decimals);
}

// State
let state = {
    tickers: [],
    contracts: [],
    selectedContracts: [], // Array of selected contracts for simulator
    loading: false,
    error: null,
    underlyingPrice: 0,
};

// AI Score Elements (Badge + Modal)
const aiScoreElements = {
    // Badge elements
    box: document.getElementById('aiScoreBox'),
    symbol: document.getElementById('aiScoreSymbol'),
    value: document.getElementById('aiScoreValue'),
    rating: document.getElementById('aiScoreRating'),
    date: document.getElementById('aiScoreDate'),
    refresh: document.getElementById('aiScoreRefresh'),
    detailsToggle: document.getElementById('aiScoreDetailsToggle'),
    // Modal elements
    modalOverlay: document.getElementById('aiScoreModalOverlay'),
    modalClose: document.getElementById('aiScoreModalClose'),
    modalSymbol: document.getElementById('aiModalSymbol'),
    modalScore: document.getElementById('aiModalScore'),
    modalRating: document.getElementById('aiModalRating'),
    modalDate: document.getElementById('aiModalDate'),
    rawScore: document.getElementById('aiScoreRaw'),
};

// Rating icons for color-blind accessibility
const RATING_ICONS = {
    'Strong Buy': '⬆️',
    'Buy': '↗️',
    'Hold': '➡️',
    'Sell': '↘️',
    'Must Sell': '⬇️',
};

// DOM Elements
const elements = {
    tickerSelect: document.getElementById('tickerSelect'),
    targetPctInput: document.getElementById('targetPct'),
    modeSelect: document.getElementById('modeSelect'),
    fetchBtn: document.getElementById('fetchBtn'),
    symbolDisplay: document.getElementById('symbolDisplay'),
    underlyingPrice: document.getElementById('underlyingPrice'),
    targetPrice: document.getElementById('targetPrice'),
    targetPctDisplay: document.getElementById('targetPctDisplay'),
    contractsBody: document.getElementById('contractsBody'),
    loadingState: document.getElementById('loadingState'),
    emptyState: document.getElementById('emptyState'),
    tableContainer: document.getElementById('tableContainer'),
    noResultsState: document.getElementById('noResultsState'),
    errorDisplay: document.getElementById('errorDisplay'),
    infoCards: document.getElementById('infoCards'),
    simulator: document.getElementById('simulator'),
    closeSimulator: document.getElementById('closeSimulator'),
    simContractDisplay: document.getElementById('simContractDisplay'),
    simStrikeDisplay: document.getElementById('simStrikeDisplay'),
    simExpirationDisplay: document.getElementById('simExpirationDisplay'),
    simPremiumDisplay: document.getElementById('simPremiumDisplay'),
    simCostDisplay: document.getElementById('simCostDisplay'),
    simContracts: document.getElementById('simContracts'),
    simUnderlying: document.getElementById('simUnderlying'),
    simUnderlyingDisplay: document.getElementById('simUnderlyingDisplay'),
    simContractsInfo: document.getElementById('simContractsInfo'),
    simTargetInput: document.getElementById('simTargetInput'),
    simResultsGrid: document.getElementById('simResultsGrid'),
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadTickers();
    setupEventListeners();
    // Fetch AI score for the initially selected ticker (SPY by default)
    if (elements.tickerSelect.value) {
        fetchAIScore(elements.tickerSelect.value);
    }
    // Auto-fetch SPY High Probability LEAPS on page load
    if (elements.tickerSelect.value === 'SPY' && elements.modeSelect.value === 'high_prob') {
        fetchLEAPS();
    }
});

// Load supported tickers
async function loadTickers() {
    try {
        const response = await fetch('/api/tickers');
        if (!response.ok) throw new Error('Failed to load tickers');

        state.tickers = await response.json();

        // Populate ticker dropdown (escape values to prevent XSS)
        elements.tickerSelect.innerHTML = state.tickers.map(t =>
            `<option value="${escapeHtml(t.symbol)}" data-target="${escapeHtml(t.default_target_pct)}">
                ${escapeHtml(t.symbol)} - ${escapeHtml(t.name)}
            </option>`
        ).join('');

        // Set default target pct from first ticker
        if (state.tickers.length > 0) {
            elements.targetPctInput.value = (state.tickers[0].default_target_pct * 100).toFixed(0);
        }
    } catch (err) {
        console.error('Error loading tickers:', err);
        showError('Failed to load supported tickers');
    }
}

// Fetch AI Score for a symbol
async function fetchAIScore(symbol, forceRefresh = false) {
    if (!symbol || !aiScoreElements.box) return;

    // Show loading state (use flex for inline layout)
    aiScoreElements.box.style.display = 'flex';
    aiScoreElements.box.classList.add('loading');
    aiScoreElements.box.classList.remove('error');
    aiScoreElements.symbol.textContent = symbol;
    aiScoreElements.value.textContent = '...';
    aiScoreElements.rating.textContent = 'Loading';
    aiScoreElements.rating.className = 'ai-score-rating';
    aiScoreElements.date.textContent = '';

    // Spin the refresh button while loading
    if (aiScoreElements.refresh) {
        aiScoreElements.refresh.classList.add('spinning');
        aiScoreElements.refresh.disabled = true;
    }

    try {
        const url = `/api/ai-score?symbol=${encodeURIComponent(symbol)}${forceRefresh ? '&refresh=true' : ''}`;
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error('Failed to fetch AI score');
        }

        const data = await response.json();
        updateAIScoreDisplay(data);

    } catch (err) {
        console.error('Error fetching AI score:', err);
        aiScoreElements.box.classList.add('error');
        aiScoreElements.value.textContent = 'N/A';
        aiScoreElements.rating.textContent = '❌ Error';
        aiScoreElements.rating.className = 'ai-score-rating';
        aiScoreElements.date.textContent = 'Click refresh to retry';
    } finally {
        aiScoreElements.box.classList.remove('loading');
        if (aiScoreElements.refresh) {
            aiScoreElements.refresh.classList.remove('spinning');
            aiScoreElements.refresh.disabled = false;
        }
    }
}

// Update AI Score display (Badge + Modal)
function updateAIScoreDisplay(data) {
    const rating = data.ai_rating;
    const ratingClass = getRatingClass(rating);
    const formattedDate = formatExactDate(data.date);
    const shortDate = formatShortDate(data.date);

    // Update Badge
    aiScoreElements.symbol.textContent = data.symbol;
    aiScoreElements.value.textContent = data.score_0_1.toFixed(2);
    aiScoreElements.date.textContent = shortDate;
    aiScoreElements.rating.textContent = rating;
    aiScoreElements.rating.className = `ai-badge-rating ${ratingClass}`;

    // Update Modal
    if (aiScoreElements.modalSymbol) {
        aiScoreElements.modalSymbol.textContent = data.symbol;
    }
    if (aiScoreElements.modalScore) {
        aiScoreElements.modalScore.textContent = data.score_0_1.toFixed(2);
    }
    if (aiScoreElements.modalRating) {
        aiScoreElements.modalRating.textContent = rating;
        aiScoreElements.modalRating.className = `ai-modal-rating ${ratingClass}`;
    }
    if (aiScoreElements.modalDate) {
        aiScoreElements.modalDate.textContent = formattedDate;
    }
    if (aiScoreElements.rawScore) {
        aiScoreElements.rawScore.textContent = data.score_0_1.toFixed(4);
    }
}

// Get CSS class for rating
function getRatingClass(rating) {
    switch (rating) {
        case 'Strong Buy': return 'strong-buy';
        case 'Buy': return 'buy';
        case 'Hold': return 'hold';
        case 'Sell': return 'sell';
        case 'Must Sell': return 'must-sell';
        default: return 'hold';
    }
}

// Format date for badge (short format)
function formatShortDate(dateStr) {
    const date = new Date(dateStr);
    const month = date.toLocaleDateString('en-US', { month: 'short' });
    const day = date.getDate();
    return `${month} ${day}`;
}

// Format date as relative time (e.g., "Updated today", "Updated yesterday")
function formatExactDate(dateStr) {
    const date = new Date(dateStr);
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return `Updated: ${date.toLocaleDateString('en-US', options)}`;
}

// Setup event listeners
function setupEventListeners() {
    // Ticker change - update default target, fetch AI score, and clear old results
    elements.tickerSelect.addEventListener('change', () => {
        const selected = elements.tickerSelect.selectedOptions[0];
        if (selected) {
            const targetPct = parseFloat(selected.dataset.target);
            elements.targetPctInput.value = (targetPct * 100).toFixed(0);
            // Clear old results and simulator
            clearResultsAndSimulator();
            // Fetch AI score for the new ticker
            fetchAIScore(selected.value);
        }
    });

    // Mode change - clear old results and simulator
    elements.modeSelect.addEventListener('change', () => {
        clearResultsAndSimulator();
    });

    // Fetch button
    elements.fetchBtn.addEventListener('click', fetchLEAPS);

    // Close simulator button
    if (elements.closeSimulator) {
        elements.closeSimulator.addEventListener('click', closeSimulator);
    }

    // AI Score refresh button
    if (aiScoreElements.refresh) {
        aiScoreElements.refresh.addEventListener('click', () => {
            const symbol = elements.tickerSelect.value;
            if (symbol) {
                fetchAIScore(symbol, true); // Force refresh
            }
        });
    }

    // AI Score Details button - opens modal
    if (aiScoreElements.detailsToggle && aiScoreElements.modalOverlay) {
        aiScoreElements.detailsToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            openAIModal();
        });
    }

    // AI Modal close button
    if (aiScoreElements.modalClose) {
        aiScoreElements.modalClose.addEventListener('click', closeAIModal);
    }

    // Close modal when clicking overlay
    if (aiScoreElements.modalOverlay) {
        aiScoreElements.modalOverlay.addEventListener('click', (e) => {
            if (e.target === aiScoreElements.modalOverlay) {
                closeAIModal();
            }
        });
    }

    // Close modal with Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && aiScoreElements.modalOverlay?.style.display !== 'none') {
            closeAIModal();
        }
    });

    // Keyboard shortcut: Enter to fetch
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.target.matches('input, textarea')) {
            fetchLEAPS();
        }
    });
}

// Open AI Market View modal
function openAIModal() {
    if (aiScoreElements.modalOverlay) {
        aiScoreElements.modalOverlay.style.display = 'flex';
        document.body.style.overflow = 'hidden'; // Prevent scrolling
    }
}

// Close AI Market View modal
function closeAIModal() {
    if (aiScoreElements.modalOverlay) {
        aiScoreElements.modalOverlay.style.display = 'none';
        document.body.style.overflow = ''; // Restore scrolling
    }
}

// Close the simulator panel
function closeSimulator() {
    if (elements.simulator) {
        elements.simulator.style.display = 'none';
    }
    // Clear selection state
    state.selectedContracts = [];
    document.querySelectorAll('[data-contract].selected').forEach(row => {
        row.classList.remove('selected');
    });
    // Update AI Explainer state (hide button when no simulation)
    updateAiExplainerState();
}

// Clear results table and simulator when ticker/mode changes
function clearResultsAndSimulator() {
    // Clear state
    state.contracts = [];
    state.selectedContracts = [];

    // Hide results table
    if (elements.tableContainer) {
        elements.tableContainer.style.display = 'none';
    }

    // Hide no results state
    if (elements.noResultsState) {
        elements.noResultsState.style.display = 'none';
    }

    // Show empty state
    if (elements.emptyState) {
        elements.emptyState.style.display = 'block';
    }

    // Hide info cards
    if (elements.infoCards) {
        elements.infoCards.style.display = 'none';
    }

    // Hide simulator
    if (elements.simulator) {
        elements.simulator.style.display = 'none';
    }

    // Clear AI Explainer
    if (aiExplainerController) {
        aiExplainerController.clearExplanation();
    }
    updateAiExplainerState();
}

// Fetch LEAPS data
async function fetchLEAPS() {
    const symbol = elements.tickerSelect.value;
    const targetPct = parseFloat(elements.targetPctInput.value) / 100;
    const mode = elements.modeSelect.value;

    if (!symbol) {
        showError('Please select a ticker');
        return;
    }

    if (isNaN(targetPct) || targetPct <= 0 || targetPct > 2) {
        showError('Please enter a valid target percentage (1-200)');
        return;
    }

    setLoading(true);
    hideError();

    try {
        const response = await fetch('/api/leaps', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol,
                target_pct: targetPct,
                mode,
                top_n: 10, // Display top 10 LEAPS only
            }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to fetch LEAPS data');
        }

        const data = await response.json();
        updateUI(data);

    } catch (err) {
        console.error('Error fetching LEAPS:', err);
        showError(err.message);
    } finally {
        setLoading(false);
    }
}

// Update UI with data
function updateUI(data) {
    state.contracts = data.contracts;
    state.underlyingPrice = data.underlying_price;

    // Show info cards (use grid, not flex - matches CSS)
    if (elements.infoCards) {
        elements.infoCards.style.display = 'grid';
    }

    // Update info cards
    elements.symbolDisplay.textContent = data.symbol;
    elements.underlyingPrice.textContent = `$${data.underlying_price.toFixed(2)}`;
    elements.targetPrice.textContent = `$${data.target_price.toFixed(2)}`;
    elements.targetPctDisplay.textContent = `+${(data.target_pct * 100).toFixed(0)}%`;

    // Update simulator underlying price (display and hidden input)
    elements.simUnderlying.value = data.underlying_price.toFixed(2);
    if (elements.simUnderlyingDisplay) {
        elements.simUnderlyingDisplay.textContent = `$${data.underlying_price.toFixed(2)}`;
    }

    // Set target prices for simulation (10% to 65%, every 5%)
    const underlying = data.underlying_price;
    const targets = [];
    for (let pct = 10; pct <= 65; pct += 5) {
        targets.push((underlying * (1 + pct / 100)).toFixed(2));
    }
    elements.simTargetInput.value = targets.join(', ');

    // Update table
    if (data.contracts.length === 0) {
        elements.emptyState.style.display = 'none';
        elements.tableContainer.style.display = 'none';
        if (elements.noResultsState) {
            elements.noResultsState.style.display = 'block';
        }
        // Clear selected contracts and hide simulator when no results
        state.selectedContracts = [];
        elements.simContracts.value = '';
        if (elements.simulator) {
            elements.simulator.style.display = 'none';
        }
    } else {
        elements.emptyState.style.display = 'none';
        if (elements.noResultsState) {
            elements.noResultsState.style.display = 'none';
        }
        elements.tableContainer.style.display = 'block';
        renderTable(data.contracts);

        // Auto-select top contract and run simulator
        const topContract = data.contracts[0];
        state.selectedContracts = [topContract];
        elements.simContracts.value = topContract.contract_symbol;

        // Highlight the first row as selected
        const firstRow = document.querySelector('[data-contract="0"]');
        if (firstRow) {
            firstRow.classList.add('selected');
        }

        // Show simulator with top contract
        showSimulator(topContract);
    }
}

// Render contracts table (with XSS protection)
function renderTable(contracts) {
    elements.contractsBody.innerHTML = contracts.map((c, idx) => `
        <tr data-contract="${idx}">
            <td class="col-contract">${escapeHtml(c.contract_symbol)}</td>
            <td class="col-expiration">${escapeHtml(c.expiration)}</td>
            <td class="col-strike">$${formatNumber(c.strike, 2)}</td>
            <td class="col-premium">$${formatNumber(c.premium, 2)}</td>
            <td class="col-cost hide-mobile">$${formatNumber(c.cost, 0)}</td>
            <td class="col-payoff hide-mobile">$${formatNumber(c.payoff_target, 0)}</td>
            <td class="col-roi ${c.roi_target >= 0 ? 'positive' : 'negative'}">${formatNumber(c.roi_target, 1)}%</td>
            <td class="col-ease hide-mobile">${formatNumber(c.ease_score, 2)}</td>
            <td class="col-roi-score hide-mobile">${formatNumber(c.roi_score, 2)}</td>
            <td class="col-score"><span class="score-badge ${getScoreClass(c.score)}">${formatNumber(c.score, 2)}</span></td>
            <td class="col-iv hide-mobile">${c.implied_volatility ? formatNumber(c.implied_volatility * 100, 1) + '%' : '-'}</td>
            <td class="col-oi hide-mobile">${c.open_interest ? c.open_interest.toLocaleString() : '-'}</td>
        </tr>
    `).join('');

    // Add click handlers
    document.querySelectorAll('[data-contract]').forEach(row => {
        row.addEventListener('click', () => {
            const idx = parseInt(row.dataset.contract);
            toggleContractSelection(contracts[idx], row);
        });
    });
}

// Toggle contract selection for simulator
function toggleContractSelection(contract, row) {
    const symbol = contract.contract_symbol;
    const existingIdx = state.selectedContracts.findIndex(c => c.contract_symbol === symbol);

    // Clear all selections first
    document.querySelectorAll('[data-contract].selected').forEach(r => {
        r.classList.remove('selected');
    });

    if (existingIdx >= 0) {
        // Clicking same row - deselect and hide simulator
        state.selectedContracts = [];
        if (elements.simulator) {
            elements.simulator.style.display = 'none';
        }
    } else {
        // Select this contract (single selection)
        state.selectedContracts = [contract];
        row.classList.add('selected');

        // Show and update simulator
        showSimulator(contract);
    }

    // Update input field
    elements.simContracts.value = state.selectedContracts.map(c => c.contract_symbol).join(', ');
}

// Show the simulator panel with contract details
function showSimulator(contract) {
    if (!elements.simulator) return;

    // Show the simulator panel
    elements.simulator.style.display = 'block';

    // Update contract info displays
    if (elements.simContractDisplay) {
        elements.simContractDisplay.textContent = contract.contract_symbol;
    }
    if (elements.simStrikeDisplay) {
        elements.simStrikeDisplay.textContent = `$${formatNumber(contract.strike, 2)}`;
    }
    if (elements.simExpirationDisplay) {
        elements.simExpirationDisplay.textContent = contract.expiration;
    }
    if (elements.simPremiumDisplay) {
        elements.simPremiumDisplay.textContent = `$${formatNumber(contract.premium, 2)}`;
    }
    if (elements.simCostDisplay) {
        elements.simCostDisplay.textContent = `$${formatNumber(contract.cost, 0)}`;
    }

    // Run simulation automatically
    runSimulation();

    // Clear previous AI explanation and update state
    if (aiExplainerController) {
        aiExplainerController.clearExplanation();
    }
    updateAiExplainerState();
}

// Update contracts info display (with XSS protection)
function updateContractsInfo() {
    if (state.selectedContracts.length === 0) {
        elements.simContractsInfo.innerHTML = '';
        return;
    }

    let html = '<div class="info-cards">';
    state.selectedContracts.forEach(c => {
        const breakeven = c.strike + c.premium;
        const escapedSymbol = escapeHtml(c.contract_symbol);
        html += `
            <div class="info-card contract-card">
                <button class="remove-contract" data-symbol="${escapedSymbol}" aria-label="Remove contract">&times;</button>
                <div class="info-card-label">${escapedSymbol}</div>
                <div style="font-size: 0.875rem; margin-top: 0.5rem;">
                    <div>Strike: <strong>$${formatNumber(c.strike, 2)}</strong></div>
                    <div>Premium: <strong>$${formatNumber(c.premium, 2)}</strong></div>
                    <div>Cost: <strong>$${formatNumber(c.cost, 0)}</strong></div>
                    <div style="color: var(--primary-color);">Breakeven: <strong>$${formatNumber(breakeven, 2)}</strong></div>
                </div>
            </div>
        `;
    });
    html += '</div>';
    elements.simContractsInfo.innerHTML = html;

    // Add remove button handlers
    document.querySelectorAll('.remove-contract').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const symbol = btn.dataset.symbol;
            const idx = state.selectedContracts.findIndex(c => c.contract_symbol === symbol);
            if (idx >= 0) {
                state.selectedContracts.splice(idx, 1);
                // Update row selection
                document.querySelectorAll('[data-contract]').forEach(row => {
                    const rowIdx = parseInt(row.dataset.contract);
                    if (state.contracts[rowIdx]?.contract_symbol === symbol) {
                        row.classList.remove('selected');
                    }
                });
                elements.simContracts.value = state.selectedContracts.map(c => c.contract_symbol).join(', ');
                updateContractsInfo();
            }
        });
    });
}

// Get score class for styling
function getScoreClass(score) {
    if (score >= 0.7) return 'score-high';
    if (score >= 0.4) return 'score-medium';
    return 'score-low';
}

// Run ROI simulation
async function runSimulation() {
    const underlying = parseFloat(elements.simUnderlying.value);
    const targetStr = elements.simTargetInput.value;

    // Get contracts either from state or parse from input
    let contractsToSimulate = state.selectedContracts;

    // If no contracts selected but input has values, try to match from loaded contracts
    if (contractsToSimulate.length === 0) {
        const inputSymbols = elements.simContracts.value.split(',').map(s => s.trim()).filter(s => s);
        if (inputSymbols.length > 0) {
            contractsToSimulate = inputSymbols.map(symbol => {
                const found = state.contracts.find(c => c.contract_symbol === symbol);
                if (found) return found;
                // If not found, we can't simulate without strike/premium
                return null;
            }).filter(c => c !== null);

            if (contractsToSimulate.length === 0) {
                showError('Could not find contract data. Please select contracts from the table.');
                return;
            }
        }
    }

    if (contractsToSimulate.length === 0) {
        showError('Please select at least one contract to simulate');
        return;
    }

    if (isNaN(underlying) || underlying <= 0) {
        showError('Please enter a valid underlying price');
        return;
    }

    // Parse target prices
    const targets = targetStr.split(',')
        .map(s => parseFloat(s.trim()))
        .filter(n => !isNaN(n) && n > 0);

    if (targets.length === 0) {
        showError('Please enter at least one target price');
        return;
    }

    // Simulate for all contracts
    renderMultiContractResults(contractsToSimulate, underlying, targets);
}

// Render simulation results for multiple contracts (with XSS protection and mobile support)
function renderMultiContractResults(contracts, underlying, targets) {
    // On mobile, show fewer columns (every 10% instead of 5%)
    const isMobile = window.innerWidth < 768;
    const filteredTargets = isMobile
        ? targets.filter((_, idx) => idx % 2 === 0)  // Show 0%, 10%, 20%, 30%, 40%, 50%, 60%
        : targets;

    // Build table header with target percentages
    let headerHtml = '<th class="sim-contract-col">Contract</th>';
    filteredTargets.forEach(target => {
        const pct = Math.round(((target - underlying) / underlying) * 100);
        headerHtml += `<th class="sim-pct-col">${pct}%</th>`;
    });

    let rowsHtml = '';
    contracts.forEach(contract => {
        const strike = contract.strike;
        const premium = contract.premium;
        const cost = premium * 100;
        const breakeven = strike + premium;
        const escapedSymbol = escapeHtml(contract.contract_symbol);

        // Contract info cell
        rowsHtml += `
            <tr>
                <td class="sim-contract-cell">
                    <div class="sim-contract-name">${escapedSymbol}</div>
                    <div class="sim-contract-details">
                        Strike: $${formatNumber(strike, 0)} | Cost: $${formatNumber(cost, 0)}
                    </div>
                    <div class="sim-contract-breakeven">
                        BE: $${formatNumber(breakeven, 2)}
                    </div>
                </td>
        `;

        // ROI cells for each target
        filteredTargets.forEach(target => {
            const intrinsic = Math.max(target - strike, 0);
            const payoff = intrinsic * 100;
            const profit = payoff - cost;
            const roiPct = (profit / cost) * 100;
            const pctMove = Math.round(((target - underlying) / underlying) * 100);

            const colorClass = roiPct >= 0 ? 'positive' : 'negative';
            rowsHtml += `
                <td class="sim-roi-cell" data-pct="+${pctMove}%">
                    <div class="sim-roi-value ${colorClass}">
                        ${roiPct >= 0 ? '+' : ''}${formatNumber(roiPct, 0)}%
                    </div>
                    <div class="sim-roi-profit">
                        ${profit >= 0 ? '+' : ''}$${formatNumber(profit, 0)}
                    </div>
                </td>
            `;
        });

        rowsHtml += '</tr>';
    });

    elements.simResultsGrid.innerHTML = `
        <div class="table-container sim-results-table">
            <table>
                <thead>
                    <tr>${headerHtml}</tr>
                </thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        </div>
    `;
}

// Set loading state
function setLoading(loading) {
    state.loading = loading;
    elements.fetchBtn.disabled = loading;
    elements.loadingState.style.display = loading ? 'flex' : 'none';

    if (loading) {
        elements.tableContainer.style.display = 'none';
        elements.emptyState.style.display = 'none';
        if (elements.noResultsState) {
            elements.noResultsState.style.display = 'none';
        }
        if (elements.simulator) {
            elements.simulator.style.display = 'none';
        }
    }
}

// Show error message
function showError(message) {
    // Find the error message span inside the error banner
    const errorMsg = elements.errorDisplay.querySelector('.error-message');
    if (errorMsg) {
        errorMsg.textContent = message;
    } else {
        elements.errorDisplay.textContent = message;
    }
    elements.errorDisplay.style.display = 'flex';
}

// Hide error message
function hideError() {
    elements.errorDisplay.style.display = 'none';
}

// =====================================================
// AI Explainer Integration
// =====================================================

// Global reference to AI Explainer controller
let aiExplainerController = null;

// Get current simulation metadata for AI Explainer
function getSimulationMetadata() {
    // Check if we have selected contracts
    if (state.selectedContracts.length === 0) {
        return null;
    }

    const contract = state.selectedContracts[0];
    const underlying = parseFloat(elements.simUnderlying.value);
    const symbol = elements.tickerSelect.value;
    const mode = elements.modeSelect.value;

    // Calculate key metrics
    const strike = contract.strike;
    const premium = contract.premium;
    const cost = premium * 100;
    const breakeven = strike + premium;
    const breakevenPct = ((breakeven / underlying) - 1) * 100;

    // Calculate ROI at various target levels
    const targetPcts = [10, 20, 30, 40, 50];
    const roiByTarget = {};
    targetPcts.forEach(pct => {
        const targetPrice = underlying * (1 + pct / 100);
        const intrinsic = Math.max(targetPrice - strike, 0);
        const payoff = intrinsic * 100;
        const profit = payoff - cost;
        const roi = (profit / cost) * 100;
        roiByTarget[`+${pct}%`] = {
            target_price: parseFloat(targetPrice.toFixed(2)),
            roi: parseFloat(roi.toFixed(1)),
            profit: parseFloat(profit.toFixed(0))
        };
    });

    // Build metadata object
    return {
        symbol: symbol,
        underlying_price: underlying,
        scoring_mode: mode,
        contract: {
            symbol: contract.contract_symbol,
            expiration: contract.expiration,
            strike: strike,
            premium: premium,
            cost: cost,
            breakeven: parseFloat(breakeven.toFixed(2)),
            breakeven_pct: parseFloat(breakevenPct.toFixed(1)),
            implied_volatility: contract.implied_volatility ? parseFloat((contract.implied_volatility * 100).toFixed(1)) : null,
            open_interest: contract.open_interest || null,
            ease_score: contract.ease_score ? parseFloat(contract.ease_score.toFixed(2)) : null,
            roi_score: contract.roi_score ? parseFloat(contract.roi_score.toFixed(2)) : null,
            composite_score: contract.score ? parseFloat(contract.score.toFixed(2)) : null
        },
        roi_simulation: roiByTarget,
        context: {
            target_pct_config: parseFloat(elements.targetPctInput.value),
            contracts_analyzed: state.contracts.length
        }
    };
}

// Initialize AI Explainer after DOM is ready
function initAiExplainer() {
    // Check if AiExplainerController is available
    if (typeof AiExplainerController === 'undefined') {
        console.warn('AiExplainerController not available');
        return;
    }

    // Initialize the controller
    aiExplainerController = new AiExplainerController({
        buttonContainerId: 'aiExplainerBtnContainer',
        panelContainerId: 'aiExplainerPanelContainer',
        pageId: 'leaps_ranker',
        contextType: 'roi_simulator',
        getMetadata: getSimulationMetadata
    });

    // Render the button (initially may be disabled if no simulation)
    aiExplainerController.render();
}

// Re-render AI Explainer button when simulation state changes
function updateAiExplainerState() {
    if (aiExplainerController) {
        aiExplainerController.render();
    }
}

// Add AI Explainer initialization to DOMContentLoaded
const originalDOMContentLoaded = document.addEventListener;
document.addEventListener('DOMContentLoaded', () => {
    // Initialize AI Explainer after a short delay to ensure DOM is ready
    setTimeout(initAiExplainer, 100);
});
