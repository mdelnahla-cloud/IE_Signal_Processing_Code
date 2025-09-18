%% Impact-Echo Delamination Estimation for Bridge Decks
% Authors: Mahmoud Elnahla, Yanlin Guo, Rebecca Atadero
% Based on: Sengupta, A., Guler, S. I., & Shokouhi, P. (2021).
% "Interpreting Impact Echo Data to Predict Condition Rating of Concrete Bridge Decks: A Machine-Learning Approach"
% Journal of Bridge Engineering, 26(8), 04021044. https://doi.org/10.1061/(ASCE)BE.1943-5592.0001744
%
% GOAL: Estimate % delamination in a concrete bridge deck using raw Impact-Echo (IE) signals.
%       Delamination % = % of test points classified as "Poor" (low-frequency energy → shallow delamination)
%
% INPUT FILE FORMAT:
%   - Excel file (e.g., "input_IE_data.xlsx") with columns:
%       Column 1: Y coordinate (ft) — optional, for plotting only
%       Column 2: X coordinate (ft) — optional, for plotting only
%       Columns 3:end: Raw IE time-domain signals (each row = one test point)
%   - Sampling frequency: 192 kHz (adjust 'fs' below if different)
%   - Deck thickness: 8.5 inches (adjust 'Deck_thickness_in' below if different)
%
% OUTPUT:
%   - Plots: Test grid, GMM fit, 3D band distribution, final delamination pie chart
%   - Console output: Final delamination percentage
%
% METHOD:
%   1. Preprocess signals (detrend, filter <60kHz)
%   2. Compute FFT → find dominant peak frequency per signal
%   3. Fit GMM → estimate "thickness frequency" (μ ± σ) from peak frequencies
%   4. Split spectrum into 3 bands: Low (<μ−σ), Mid (μ±σ), High (>μ+σ)
%   5. Compute % energy in each band → 3D point per signal
%   6. Classify using published Cp centroids (Sengupta et al. 2021) → "Poor", "Good", "Fair"
%   7. Report % "Poor" → estimated delamination %
%
% NOTE:
%   - Filter: FIR Equiripple LPF @ 60kHz (designed internally — see Step 2)
%   - Cp: Published k-means centroids from Sengupta et al. (2021).
%   - GMM K: Selected by BIC elbow — user visually confirms (default K=3)

clear; clc; close all;
fprintf('Starting Impact-Echo Delamination Estimation...\n');

%% ==================== USER CONFIGURATION ====================

% ⚙️ Adjust these if your data differs
input_file = 'input_IE_data.xlsx';   % Your IE data file (see format above)
fs = 192e3;                         % Sampling frequency (Hz) — adjust if needed
Deck_thickness_in = 8.5;            % Deck thickness in inches — adjust if needed

%% ==================== DATA LOADING ====================

if ~exist(input_file, 'file')
    error('Input file "%s" not found. Please provide IE data in specified format.', input_file);
end

data_table = readtable(input_file);
signal_data = table2array(data_table(:, 3:end));  % Columns 3+ = raw IE signals
fprintf('Loaded %d signals from "%s".\n', size(signal_data, 1), input_file);

% Optional: Extract coordinates for plotting (if available)
if size(data_table, 2) >= 2
    X_coords = table2array(data_table(:, 2));
    Y_coords = table2array(data_table(:, 1));
else
    X_coords = 1:size(signal_data, 1);
    Y_coords = ones(size(signal_data, 1), 1);
end

% Plot test grid (if coordinates available)
figure('Name', 'Test Locations');
plot(X_coords, Y_coords, 'r.', 'MarkerSize', 10);
title('IE Test Grid', 'FontWeight', 'bold');
xlabel('Length (ft)'); ylabel('Width (ft)');
grid on;

%% ==================== SIGNAL SETUP ====================

N = size(signal_data, 2);           % Samples per signal
f = fs * (0 : N/2 - 1) / N;         % One-sided frequency vector (Hz)

%% ==================== DETREND SIGNALS ====================

signals_detrended = signal_data - mean(signal_data, 2);

%% ==================== DESIGN & APPLY LOWPASS FILTER ====================
% FIR Equiripple LPF: Fpass=60kHz, Fstop=61kHz, Apass=1dB, Astop=80dB
% Matches Filter Designer screenshot — sharp transition to preserve <60kHz

fprintf('Designing and applying lowpass filter (FIR Equiripple, 60→61 kHz)...\n');
Hd = designfilt('lowpassfir', 'SampleRate', fs, ...
    'PassbandFrequency', 60e3, 'StopbandFrequency', 61e3, ...
    'PassbandRipple', 1, 'StopbandAttenuation', 80, ...
    'DesignMethod', 'equiripple');

% Optional: Plot filter response
figure('Name', 'Filter Response');
freqz(Hd, 1024, fs);
title('Designed Lowpass Filter (FIR Equiripple)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;

% Apply filter
filtered_signals = filter(Hd, signals_detrended');
filtered_signals = filtered_signals'; % Back to row-per-signal

%% ==================== FFT & POWER SPECTRUM ====================

fft_all = fft(filtered_signals, [], 2);
fft_onesided = fft_all(:, 1:N/2);
power_spectrum = abs(fft_onesided).^2;

%% ==================== PEAK DETECTION & FILTER >25KHZ ====================

fprintf('Detecting dominant peak frequencies...\n');
peak_freqs = [];
for i = 1:size(power_spectrum, 1)
    [vals, locs] = findpeaks(power_spectrum(i, :));
    if ~isempty(vals)
        [~, idx] = max(vals);
        peak_freqs = [peak_freqs, f(locs(idx))];
    end
end
peak_freqs = peak_freqs(peak_freqs <= 25e3); 
fprintf('Detected %d valid peak frequencies (≤25 kHz).\n', length(peak_freqs));

%% ==================== GMM: ESTIMATE THICKNESS FREQUENCY ====================

maxK = 4; BIC_scores = zeros(1, maxK);
for k = 1:maxK
    gm = fitgmdist(peak_freqs', k, 'RegularizationValue', 1e-6);
    BIC_scores(k) = gm.BIC;
end

% Plot BIC — USER MUST VISUALLY SELECT ELBOW
figure('Name', 'GMM Model Selection (BIC)');
plot(1:maxK, BIC_scores, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Components (K)'); ylabel('BIC Score');
title('Select K at "Elbow" (Point of Diminishing Returns)');
grid on;

% Default to K=3 (as in paper) if no user input
prompt = 'Enter K (1-4) based on BIC elbow [default=3]: ';
K = input(prompt, 's');
if isempty(K) || ~isnumeric(str2double(K))
    K = 3;
    fprintf('Using default K = %d (Sengupta et al. 2021).\n', K);
else
    K = str2double(K);
end

% Fit final GMM
gm_final = fitgmdist(peak_freqs', K, 'RegularizationValue', 1e-6);
mus = gm_final.mu; sigmas = sqrt(gm_final.Sigma); weights = gm_final.ComponentProportion;

% Thickness component = highest weight (paper: "most prevalent frequency")
[~, thick_idx] = max(weights);
mu_thick = mus(thick_idx); sigma_thick = sigmas(thick_idx);

fprintf('Thickness Freq: %.1f Hz (σ=%.1f Hz) — Component %d (weight=%.2f)\n', ...
    mu_thick, sigma_thick, thick_idx, weights(thick_idx));


%% ==================== PLOT HISTOGRAM + VALIDATE WITH THEORETICAL RANGE ====================

% Plot histogram for peak frequencies (Figure 1)
figure('Name', 'Peak Frequency Histogram with Validation');
h = histogram(peak_freqs, 'BinWidth', 0.48e3, 'FaceColor', [0.8 0.8 1], 'EdgeColor', 'b');
hold on;

% GMM-estimated thickness frequency range (±1σ)
xline([mu_thick - sigma_thick, mu_thick + sigma_thick], '--k', 'LineWidth', 2, 'Label', 'GMM Thickness Range (±1σ)');

% Theoretical thickness frequency range
h_m = Deck_thickness_in * 0.0254;  % Convert deck thickness to meters
Vp_low = 3500;  % Lower bound for healthy concrete
Vp_high = 4500;  % Upper bound
f_theory_low = 0.96 * Vp_low / (2 * h_m);
f_theory_high = 0.96 * Vp_high / (2 * h_m);

xline([f_theory_low, f_theory_high], '-.r', 'LineWidth', 2, 'Label', 'Theoretical Range (Vp=3500-4500 m/s)');

% Add title and labels for figure 1
xlabel('Frequency (Hz)', 'FontWeight', 'bold');
ylabel('Count', 'FontWeight', 'bold');
title('Peak Frequency Distribution with GMM and Theoretical Thickness Frequency Ranges', 'FontWeight', 'bold');
xlim([0 25e3]);
grid on;

% Correct legend for figure 1 (Peak Frequency Histogram + GMM)
legend('Peak Frequency Count', 'GMM Thickness Range (±1σ)', 'Theoretical Range (Vp=3500-4500 m/s)', 'Location', 'northeast');

% Validation Output for Figure 1
fprintf('\n--- VALIDATION: GMM vs THEORETICAL THICKNESS FREQUENCY ---\n');
fprintf('GMM Range: [%.1f, %.1f] Hz\n', mu_thick - sigma_thick, mu_thick + sigma_thick);
fprintf('Theory Range: [%.1f, %.1f] Hz\n', f_theory_low, f_theory_high);

% Check overlap between ranges
overlap = max(0, min(mu_thick + sigma_thick, f_theory_high) - max(mu_thick - sigma_thick, f_theory_low));
if overlap > 0
    fprintf('✅ VALIDATION PASSED: GMM range overlaps with theoretical range by %.1f Hz\n', overlap);
else
    fprintf('⚠️ VALIDATION WARNING: No overlap between GMM and theoretical ranges.\n');
end
fprintf('----------------------------------------------------------------\n');

% Plot histogram and GMM components (Figure 2)
figure('Name', 'Peak Freq Histogram + GMM');
histogram(peak_freqs, 'BinWidth', 0.48e3, 'Normalization', 'pdf');
hold on;

% Create and plot GMM components
x_pdf = linspace(0, 25e3, 1000)';
total_pdf = zeros(size(x_pdf));
colors = lines(K);  % Colors for GMM components
for k = 1:K
    comp_pdf = weights(k) * normpdf(x_pdf, mus(k), sigmas(k));
    plot(x_pdf, comp_pdf, 'Color', colors(k,:), 'LineWidth', 1.5);
    total_pdf = total_pdf + comp_pdf;
end

% Add title and labels for figure 2
xlabel('Frequency (Hz)', 'FontWeight', 'bold');
ylabel('Density', 'FontWeight', 'bold');
title('Peak Frequency Histogram with GMM Components', 'FontWeight', 'bold');
xlim([0 25e3]);
grid on;

% Correct legend for figure 2 (Peak Frequency + GMM components)
legend(arrayfun(@(k) sprintf('GMM Component %d', k), 1:K, 'UniformOutput', false), 'Location', 'northeast');

%% ==================== ENERGY BANDS (p1, p2, p3) ====================

fprintf('Computing spectral energy bands...\n');
band_proportions = zeros(size(power_spectrum, 1), 3);
for i = 1:size(power_spectrum, 1)
    cumE = cumtrapz(f, power_spectrum(i, :));
    getE = @(a,b) max(cumE(f<=b)) - min(cumE(f>=a));
    
    E_low  = getE(0, mu_thick - sigma_thick);
    E_mid  = getE(mu_thick - sigma_thick, mu_thick + sigma_thick);
    E_high = getE(mu_thick + sigma_thick, f(end));
    totalE = E_low + E_mid + E_high;
    
    if totalE > 0
        band_proportions(i, :) = [E_low, E_mid, E_high] / totalE;
    else
        band_proportions(i, :) = [1 1 1]/3; % fallback
    end
end

%% ==================== 3D BAND PLOT ====================

figure('Name', 'Band Energy Proportions');
plot3(band_proportions(:,1), band_proportions(:,2), band_proportions(:,3), 'b.', 'MarkerSize', 10);
xlabel('Band 1: Low (Delamination)'); set(gca, 'XDir', 'reverse');
ylabel('Band 2: Mid (Thickness)');     set(gca, 'YDir', 'reverse');
zlabel('Band 3: High (Marginal)');
title('Spectral Energy Distribution');
grid on;

%% ==================== CLASSIFICATION USING PUBLISHED Cp ====================
% From Sengupta et al. (2021): Learned via k-means on 160 bridge datasets
% Cp rows = [Poor; Good; Fair] — energy proportions [p1, p2, p3]
Cp = [0.691  0.246  0.063;   % Poor  → High Low-band (delamination)
      0.175  0.720  0.105;   % Good  → High Mid-band (intact)
      0.230  0.273  0.497];  % Fair  → High High-band (marginal/deep delam)

class_labels = {'Poor (Delamination)', 'Good (Intact)', 'Fair (Marginal)'};
class_indices = zeros(size(band_proportions, 1), 1);

for i = 1:size(band_proportions, 1)
    class_indices(i) = knnsearch(Cp, band_proportions(i, :), 'Distance', 'euclidean');
end

% Count & percentage
N_poor  = sum(class_indices == 1);
N_good  = sum(class_indices == 2);
N_fair  = sum(class_indices == 3);
totalN  = N_poor + N_good + N_fair;

pct_poor = N_poor / totalN * 100;  % ← THIS IS THE DELAMINATION ESTIMATE
pct_good = N_good / totalN * 100;
pct_fair = N_fair / totalN * 100;

%% ==================== FINAL OUTPUT: DELAMINATION % ====================

fprintf('\n');
fprintf('=================================================\n');
fprintf(' BRIDGE DECK DELAMINATION ESTIMATE\n');
fprintf(' Authors: Mahmoud ElNahla, Yanlin Guo, Rebecca Atadero\n');
fprintf(' Based on: Sengupta et al. (2021), J. Bridge Eng.\n');
fprintf('=================================================\n');
fprintf(' Total Test Points: %d\n', totalN);
fprintf(' "Poor"  (Delamination) : %d (%.1f%%) ← ESTIMATED DELAMINATION PERCENTAGE\n', N_poor, pct_poor);
fprintf(' "Good"  (Intact)       : %d (%.1f%%)\n', N_good, pct_good);
fprintf(' "Fair"  (Marginal)     : %d (%.1f%%)\n', N_fair, pct_fair);
fprintf(' Thickness Frequency    : %.1f ± %.1f Hz (GMM, K=%d)\n', mu_thick, sigma_thick, K);
fprintf('=================================================\n');

% Optional: Pie chart
figure('Name', 'Delamination Estimate');
pie([N_poor, N_good, N_fair], class_labels);
title(sprintf('Estimated Delamination: %.1f%%', pct_poor));

fprintf('✅ Delamination estimation complete.\n');