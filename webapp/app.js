// Spatial Multi-Armed Bandit Web App
// Implements make_correlated_dog reward landscape generation

class GPEnvironment {
    constructor(gridSize, lengthScale, seed = null) {
        this.gridSize = gridSize;
        this.lengthScale = lengthScale;
        this.rng = seed ? this.seededRandom(seed) : Math.random;
        this.rewardMap = null;
        this.minCoords = null;
    }

    // Simple seeded random number generator (Mulberry32)
    seededRandom(seed) {
        return function() {
            let t = seed += 0x6D2B79F5;
            t = Math.imul(t ^ t >>> 15, t | 1);
            t ^= t + Math.imul(t ^ t >>> 7, t | 61);
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
    }

    // Box-Muller transform for normal distribution
    randomNormal() {
        let u = 0, v = 0;
        while (u === 0) u = this.rng();
        while (v === 0) v = this.rng();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    // RBF kernel function
    rbfKernel(x1, y1, x2, y2) {
        const dx = x1 - x2;
        const dy = y1 - y2;
        const distSq = dx * dx + dy * dy;
        return Math.exp(-distSq / (2 * this.lengthScale * this.lengthScale));
    }

    // Generate covariance matrix using RBF kernel
    buildCovarianceMatrix() {
        const n = this.gridSize * this.gridSize;
        const K = new Array(n);
        
        for (let i = 0; i < n; i++) {
            K[i] = new Array(n);
            const xi = i % this.gridSize;
            const yi = Math.floor(i / this.gridSize);
            
            for (let j = 0; j < n; j++) {
                const xj = j % this.gridSize;
                const yj = Math.floor(j / this.gridSize);
                
                if (i === j) {
                    K[i][j] = 1.0 + 1e-6; // Add small jitter for numerical stability
                } else {
                    K[i][j] = this.rbfKernel(xi, yi, xj, yj);
                }
            }
        }
        
        return K;
    }

    // Cholesky decomposition
    cholesky(matrix) {
        const n = matrix.length;
        const L = new Array(n);
        
        for (let i = 0; i < n; i++) {
            L[i] = new Array(n).fill(0);
        }
        
        for (let i = 0; i < n; i++) {
            for (let j = 0; j <= i; j++) {
                let sum = 0;
                
                if (j === i) {
                    for (let k = 0; k < j; k++) {
                        sum += L[j][k] * L[j][k];
                    }
                    L[i][j] = Math.sqrt(Math.max(matrix[i][i] - sum, 1e-10));
                } else {
                    for (let k = 0; k < j; k++) {
                        sum += L[i][k] * L[j][k];
                    }
                    L[i][j] = (matrix[i][j] - sum) / L[j][j];
                }
            }
        }
        
        return L;
    }

    // Generate GP sample using Cholesky
    sampleGP() {
        const n = this.gridSize * this.gridSize;
        
        // For large grids, use a simpler approximation
        if (n > 400) {
            return this.normalizeGrid(this.sampleGPSimple());
        }
        
        // Build covariance matrix
        const K = this.buildCovarianceMatrix();
        
        // Cholesky decomposition
        const L = this.cholesky(K);
        
        // Generate standard normal samples
        const z = new Array(n);
        for (let i = 0; i < n; i++) {
            z[i] = this.randomNormal();
        }
        
        // Multiply by Cholesky factor
        const sample = new Array(n);
        for (let i = 0; i < n; i++) {
            sample[i] = 0;
            for (let j = 0; j <= i; j++) {
                sample[i] += L[i][j] * z[j];
            }
        }
        
        // Reshape to 2D grid and normalize to [0,1]
        const grid = new Array(this.gridSize);
        for (let i = 0; i < this.gridSize; i++) {
            grid[i] = new Array(this.gridSize);
            for (let j = 0; j < this.gridSize; j++) {
                grid[i][j] = sample[i * this.gridSize + j];
            }
        }
        
        return this.normalizeGrid(grid);
    }

    // Simpler GP sampling for large grids (using spatial smoothing)
    sampleGPSimple() {
        const grid = new Array(this.gridSize);
        
        // Generate white noise
        for (let i = 0; i < this.gridSize; i++) {
            grid[i] = new Array(this.gridSize);
            for (let j = 0; j < this.gridSize; j++) {
                grid[i][j] = this.randomNormal();
            }
        }
        
        // Apply spatial smoothing to create correlation
        const smoothed = new Array(this.gridSize);
        const kernelRadius = Math.max(1, Math.floor(this.lengthScale / 2));
        
        for (let i = 0; i < this.gridSize; i++) {
            smoothed[i] = new Array(this.gridSize);
            for (let j = 0; j < this.gridSize; j++) {
                let sum = 0;
                let weightSum = 0;
                
                for (let di = -kernelRadius; di <= kernelRadius; di++) {
                    for (let dj = -kernelRadius; dj <= kernelRadius; dj++) {
                        const ni = i + di;
                        const nj = j + dj;
                        
                        if (ni >= 0 && ni < this.gridSize && nj >= 0 && nj < this.gridSize) {
                            const dist = Math.sqrt(di * di + dj * dj);
                            const weight = Math.exp(-dist * dist / (2 * this.lengthScale * this.lengthScale));
                            sum += grid[ni][nj] * weight;
                            weightSum += weight;
                        }
                    }
                }
                
                smoothed[i][j] = sum / weightSum;
            }
        }
        
        return smoothed;
    }

    // DoG (Difference of Gaussians) kernel - matches Python's dog_rbf_landscape
    dogKernel(x, y, centerX, centerY, sigmaInner, sigmaOuter) {
        const dx = x - centerX;
        const dy = y - centerY;
        const r2 = dx * dx + dy * dy;
        
        const inner = Math.exp(-r2 / (2 * sigmaInner * sigmaInner));
        const outer = Math.exp(-r2 / (2 * sigmaOuter * sigmaOuter));
        
        // Python uses: dog = inner - outer * (sigma_inner/sigma_outer)
        return inner - outer * (sigmaInner / sigmaOuter);
    }

    // Find minimum coordinates in a grid
    findMinCoords(grid) {
        let minVal = Infinity;
        let minX = 0, minY = 0;
        
        for (let i = 0; i < grid.length; i++) {
            for (let j = 0; j < grid[i].length; j++) {
                if (grid[i][j] < minVal) {
                    minVal = grid[i][j];
                    minX = j;
                    minY = i;
                }
            }
        }
        
        return { x: minX, y: minY, value: minVal };
    }

    // Min-max normalization
    normalizeGrid(grid) {
        let minVal = Infinity;
        let maxVal = -Infinity;
        
        for (let i = 0; i < grid.length; i++) {
            for (let j = 0; j < grid[i].length; j++) {
                minVal = Math.min(minVal, grid[i][j]);
                maxVal = Math.max(maxVal, grid[i][j]);
            }
        }
        
        const range = maxVal - minVal;
        if (range === 0) return grid;
        
        const normalized = new Array(grid.length);
        for (let i = 0; i < grid.length; i++) {
            normalized[i] = new Array(grid[i].length);
            for (let j = 0; j < grid[i].length; j++) {
                normalized[i][j] = (grid[i][j] - minVal) / range;
            }
        }
        
        return normalized;
    }

    // Generate the correlated dog environment
    // Follows Python's make_correlated_dog logic:
    // 1. Sample GP landscape
    // 2. Find global minimum of GP
    // 3. Add DoG centered at GP minimum (creates peak at that location)
    // 4. Normalize so DoG center = 1, original GP max ~0.7
    generate() {
        console.log('Generating environment with gridSize:', this.gridSize, 'lengthScale:', this.lengthScale);
        
        // Step 1: Sample GP (normalized to [0,1] internally)
        const gp = this.sampleGP();
        console.log('GP sampled, range:', 
            Math.min(...gp.flat()).toFixed(3), 'to', 
            Math.max(...gp.flat()).toFixed(3));
        
        // Step 2: Find global minimum of GP (this will become the peak after DoG)
        const minInfo = this.findMinCoords(gp);
        this.minCoords = { x: minInfo.x, y: minInfo.y };
        console.log('GP min coords:', this.minCoords, 'value:', minInfo.value.toFixed(3));
        
        // Step 3: Create DoG centered at GP minimum
        // sigma_outer = length_scale, sigma_inner = length_scale / 2
        const sigmaOuter = this.lengthScale;
        const sigmaInner = sigmaOuter / 2.0;
        
        // Generate DoG pattern (unnormalized)
        const dog = new Array(this.gridSize);
        for (let i = 0; i < this.gridSize; i++) {
            dog[i] = new Array(this.gridSize);
            for (let j = 0; j < this.gridSize; j++) {
                dog[i][j] = this.dogKernel(j, i, this.minCoords.x, this.minCoords.y, sigmaInner, sigmaOuter);
            }
        }
        
        // Scale DoG peak to dog_max (1.2) - this makes the center higher than GP max
        let dogMaxAbs = 0;
        for (let i = 0; i < dog.length; i++) {
            for (let j = 0; j < dog[i].length; j++) {
                dogMaxAbs = Math.max(dogMaxAbs, Math.abs(dog[i][j]));
            }
        }
        
        const dogMax = 1.2;  // Same as Python
        for (let i = 0; i < dog.length; i++) {
            for (let j = 0; j < dog[i].length; j++) {
                dog[i][j] = (dog[i][j] / dogMaxAbs) * dogMax;
            }
        }
        
        // Center around zero mean (like Python)
        let dogSum = 0;
        for (let i = 0; i < dog.length; i++) {
            for (let j = 0; j < dog[i].length; j++) {
                dogSum += dog[i][j];
            }
        }
        const dogMean = dogSum / (this.gridSize * this.gridSize);
        
        for (let i = 0; i < dog.length; i++) {
            for (let j = 0; j < dog[i].length; j++) {
                dog[i][j] -= dogMean;
            }
        }
        
        console.log('DoG range after scaling:', 
            Math.min(...dog.flat()).toFixed(3), 'to', 
            Math.max(...dog.flat()).toFixed(3));
        
        // Step 4: Combine GP + DoG and normalize to [0,1]
        // The DoG center (at GP min) becomes the peak (1)
        // The original GP max becomes ~0.7
        const combined = new Array(this.gridSize);
        for (let i = 0; i < this.gridSize; i++) {
            combined[i] = new Array(this.gridSize);
            for (let j = 0; j < this.gridSize; j++) {
                combined[i][j] = gp[i][j] + dog[i][j];
            }
        }
        
        this.rewardMap = this.normalizeGrid(combined);
        
        // Verify: the DoG center location should now have reward = 1
        const centerReward = this.rewardMap[this.minCoords.y][this.minCoords.x];
        console.log('Environment generated:');
        console.log('  DoG center (original GP min) reward:', centerReward.toFixed(3));
        console.log('  Reward range:', 
            Math.min(...this.rewardMap.flat()).toFixed(3), 'to', 
            Math.max(...this.rewardMap.flat()).toFixed(3));
        
        return this.rewardMap;
    }

    // Get reward with observation noise
    getReward(x, y, noiseLevel) {
        if (!this.rewardMap || x < 0 || x >= this.gridSize || y < 0 || y >= this.gridSize) {
            return 0;
        }
        
        const trueReward = this.rewardMap[y][x];
        const noise = this.randomNormal() * noiseLevel;
        return Math.max(0, Math.min(1, trueReward + noise));
    }
}

// Game state
let game = null;
let gridSize = 33;
let lengthScale = 4.5;
let noiseLevel = 0.1;
let revealed = [];
let lastReward = null;
let cumulativeReward = 0;
let clickCount = 0;
let cellValues = []; // Store observed values for each cell

// Initialize game
function initGame() {
    try {
        game = new GPEnvironment(gridSize, lengthScale, Date.now());
        game.generate();
        
        // Reset state
        revealed = Array(gridSize).fill(null).map(() => Array(gridSize).fill(false));
        cellValues = Array(gridSize).fill(null).map(() => Array(gridSize).fill(null));
        lastReward = null;
        cumulativeReward = 0;
        clickCount = 0;
        
        // Update UI
        updateStats();
        renderGrid();
        console.log('Game initialized successfully');
    } catch (e) {
        console.error('Error initializing game:', e);
        alert('Error initializing game: ' + e.message);
    }
}

// Render the grid
function renderGrid() {
    console.log('Rendering grid, size:', gridSize);
    const gridEl = document.getElementById('grid');
    gridEl.innerHTML = '';
    
    // Use fixed cell size, let grid overflow if needed
    const cellSize = 12;
    
    gridEl.style.gridTemplateColumns = `repeat(${gridSize}, ${cellSize}px)`;
    gridEl.style.width = `${gridSize * (cellSize + 1)}px`;
    gridEl.style.height = `${gridSize * (cellSize + 1)}px`;
    
    for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.x = x;
            cell.dataset.y = y;
            
            if (revealed[y][x]) {
                cell.classList.add('revealed');
                const reward = cellValues[y][x];
                cell.style.backgroundColor = getColorForValue(reward);
                // Display the observed value inside the cell
                cell.textContent = reward.toFixed(2);
                cell.style.color = reward > 0.5 ? '#000' : '#fff';
            } else {
                cell.classList.add('unrevealed');
            }
            
            cell.addEventListener('click', () => handleCellClick(x, y));
            gridEl.appendChild(cell);
        }
    }
    console.log('Grid rendered, cells:', gridSize * gridSize);
}

// Get color for reward value
function getColorForValue(value) {
    // Custom color scale: dark blue -> light blue -> yellow -> gold
    const colors = [
        { pos: 0, r: 26, g: 26, b: 46 },
        { pos: 0.25, r: 45, g: 74, b: 124 },
        { pos: 0.5, r: 107, g: 140, b: 206 },
        { pos: 0.75, r: 196, g: 163, b: 90 },
        { pos: 1, r: 240, g: 192, b: 64 }
    ];
    
    // Find the two colors to interpolate between
    let c1 = colors[0], c2 = colors[1];
    for (let i = 0; i < colors.length - 1; i++) {
        if (value >= colors[i].pos && value <= colors[i + 1].pos) {
            c1 = colors[i];
            c2 = colors[i + 1];
            break;
        }
    }
    
    // Interpolate
    const t = (value - c1.pos) / (c2.pos - c1.pos);
    const r = Math.round(c1.r + t * (c2.r - c1.r));
    const g = Math.round(c1.g + t * (c2.g - c1.g));
    const b = Math.round(c1.b + t * (c2.b - c1.b));
    
    return `rgb(${r}, ${g}, ${b})`;
}

// Handle cell click
function handleCellClick(x, y) {
    try {
        if (revealed[y][x]) {
            // Re-click: sample a new noisy reward from the same cell
            const reward = game.getReward(x, y, noiseLevel);
            lastReward = reward;
            cumulativeReward += reward;
            clickCount++;
            cellValues[y][x] = reward;
            updateStats();
            renderGrid();
            return;
        }
        
        // First click: get reward with noise
        const reward = game.getReward(x, y, noiseLevel);
        
        // Update state
        revealed[y][x] = true;
        cellValues[y][x] = reward;
        lastReward = reward;
        cumulativeReward += reward;
        clickCount++;
        
        // Update UI
        updateStats();
        renderGrid();
    } catch (e) {
        console.error('Error handling click:', e);
    }
}

// Update statistics display
function updateStats() {
    document.getElementById('lastReward').textContent = lastReward !== null 
        ? lastReward.toFixed(3) 
        : '-';
    document.getElementById('cumulativeReward').textContent = cumulativeReward.toFixed(2);
    document.getElementById('clickCount').textContent = clickCount;
    
    // Mean reward per click
    const meanReward = clickCount > 0 ? (cumulativeReward / clickCount).toFixed(3) : '-';
    document.getElementById('meanReward').textContent = meanReward;
    
    const totalCells = gridSize * gridSize;
    const exploredCells = revealed.flat().filter(r => r).length;
    const percent = (exploredCells / totalCells * 100).toFixed(1);
    document.getElementById('exploredPercent').textContent = percent + '%';
}

// Event listeners
document.getElementById('gridSize').addEventListener('input', (e) => {
    gridSize = parseInt(e.target.value);
    document.getElementById('gridSizeValue').textContent = gridSize;
    initGame();
});

document.getElementById('lengthScale').addEventListener('input', (e) => {
    lengthScale = parseFloat(e.target.value);
    document.getElementById('lengthScaleValue').textContent = lengthScale;
});

document.getElementById('noiseLevel').addEventListener('input', (e) => {
    noiseLevel = parseFloat(e.target.value);
    document.getElementById('noiseLevelValue').textContent = noiseLevel;
});

document.getElementById('newGame').addEventListener('click', () => {
    console.log('New game button clicked');
    initGame();
});

// Initialize on load
console.log('App initializing...');
window.addEventListener('load', initGame);