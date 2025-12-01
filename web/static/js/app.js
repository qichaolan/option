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
    errorDisplay: document.getElementById('errorDisplay'),
    simContracts: document.getElementById('simContracts'),
    simUnderlying: document.getElementById('simUnderlying'),
    simUnderlyingDisplay: document.getElementById('simUnderlyingDisplay'),
    simContractsInfo: document.getElementById('simContractsInfo'),
    simTargetInput: document.getElementById('simTargetInput'),
    runSimBtn: document.getElementById('runSimBtn'),
    simResultsGrid: document.getElementById('simResultsGrid'),
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadTickers();
    setupEventListeners();
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

// Setup event listeners
function setupEventListeners() {
    // Ticker change - update default target
    elements.tickerSelect.addEventListener('change', () => {
        const selected = elements.tickerSelect.selectedOptions[0];
        if (selected) {
            const targetPct = parseFloat(selected.dataset.target);
            elements.targetPctInput.value = (targetPct * 100).toFixed(0);
        }
    });

    // Fetch button
    elements.fetchBtn.addEventListener('click', fetchLEAPS);

    // Run simulation button
    elements.runSimBtn.addEventListener('click', runSimulation);

    // Keyboard shortcut: Enter to fetch
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.target.matches('input, textarea')) {
            fetchLEAPS();
        }
    });
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
                top_n: 20,
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

    // Update info cards
    elements.symbolDisplay.textContent = data.symbol;
    elements.underlyingPrice.textContent = `$${data.underlying_price.toFixed(2)}`;
    elements.targetPrice.textContent = `$${data.target_price.toFixed(2)}`;
    elements.targetPctDisplay.textContent = `+${(data.target_pct * 100).toFixed(0)}%`;

    // Update simulator underlying price (display and hidden input)
    elements.simUnderlying.value = data.underlying_price.toFixed(2);
    elements.simUnderlyingDisplay.textContent = `$${data.underlying_price.toFixed(2)}`;

    // Set target prices for simulation (0% to 60%, every 5%)
    const underlying = data.underlying_price;
    const targets = [];
    for (let pct = 0; pct <= 60; pct += 5) {
        targets.push((underlying * (1 + pct / 100)).toFixed(2));
    }
    elements.simTargetInput.value = targets.join(', ');

    // Update table
    if (data.contracts.length === 0) {
        elements.emptyState.style.display = 'block';
        elements.tableContainer.style.display = 'none';
    } else {
        elements.emptyState.style.display = 'none';
        elements.tableContainer.style.display = 'block';
        renderTable(data.contracts);
    }

    // Clear selected contracts
    state.selectedContracts = [];
    elements.simContracts.value = '';
    updateContractsInfo();
}

// Render contracts table (with XSS protection)
function renderTable(contracts) {
    elements.contractsBody.innerHTML = contracts.map((c, idx) => `
        <tr data-contract="${idx}">
            <td class="col-contract">${escapeHtml(c.contract_symbol)}</td>
            <td class="col-expiration">${escapeHtml(c.expiration)}</td>
            <td class="col-strike">$${formatNumber(c.strike, 2)}</td>
            <td class="col-premium">$${formatNumber(c.premium, 2)}</td>
            <td class="col-cost">$${formatNumber(c.cost, 0)}</td>
            <td class="col-payoff">$${formatNumber(c.payoff_target, 0)}</td>
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

    if (existingIdx >= 0) {
        // Remove if already selected
        state.selectedContracts.splice(existingIdx, 1);
        row.classList.remove('selected');
    } else {
        // Add if not at limit
        if (state.selectedContracts.length >= 10) {
            showError('Maximum 10 contracts allowed');
            return;
        }
        state.selectedContracts.push(contract);
        row.classList.add('selected');
    }

    // Update input field
    elements.simContracts.value = state.selectedContracts.map(c => c.contract_symbol).join(', ');
    updateContractsInfo();
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
    }
}

// Show error message
function showError(message) {
    elements.errorDisplay.textContent = message;
    elements.errorDisplay.style.display = 'block';
}

// Hide error message
function hideError() {
    elements.errorDisplay.style.display = 'none';
}
