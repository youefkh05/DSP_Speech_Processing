clear; 
clc; 
close all;

%% Problem1 Analyze the frequency domain characteristics of Rectangular, Hanning, and Hamming windows.
%% --- Parameters ---
N = 1024;   % Window length (should be large, e.g., 512 or 1024, for good frequency resolution)
% Define the relative path to the audio file
relative_audio_path = '..\Data\Test\C02n_1.wav'; 
relative_path_to_plots = '..\Data\Results\plot'; % Define the relative path from Src to the target folder

% Extract info
[Fs, bitDepth, bitRate, numChannels, totalSamples] = get_audio_info(relative_audio_path);

% Example of using the extracted data
if ~isempty(Fs)
    fprintf('\nAnalysis using Fs = %d Hz and Bit Rate = %d bps is ready.\n', Fs, bitRate);
end
%{
%% --- 1. Define Windows (Time Domain) ---

% 1. Generate Window Vectors (using your first function)
[W_Rec, W_Han, W_Ham, n] = generate_windows(N);

% 2. Plot Time Domain and capture the figure handle
time_fig_handle = plot_time_windows(n, W_Rec, W_Han, W_Ham);

% 3. Save the Time Domain figure
figure_to_png(time_fig_handle, 'problem1_time_domain',relative_path_to_plots); 

%% --- 2. Define Windows (Frequency Domain) ---
% --- 1. Compute Frequency Responses (Log Magnitude dB) ---
[f, dB_Rec_shifted, dB_Han_shifted, dB_Ham_shifted] = compute_freq_response(Fs, W_Rec, W_Han, W_Ham);

% --- 2. Plotting ---
freq_fig_handle = plot_freq_windows(Fs, f, dB_Rec_shifted, dB_Han_shifted, dB_Ham_shifted);

% 3. Save the Frequency Domain figure
figure_to_png(freq_fig_handle, 'problem1_freq_domain',relative_path_to_plots); 
%}
%% Problem2 LPC Analysis based on given R data
% --- 1. Define Input Data for Problem 2 ---

% R_vector = [R(0), R(1), R(2), ..., R(p)]
R_vector_p2 = [1, 0.7, 0.4];
LPC_ORDER_p2 = length(R_vector_p2) - 1; % Should be 2

fprintf('--- Problem 2: LPC (p=2) Solution Verification ---\n');

% --- 2. Call the generalized solver function  ---
[a_coeffs, E_error, p_order] = lpc_matrix_solution(R_vector_p2);

% --- 3. Display Results ---
print_lpc_results(a_coeffs, E_error, p_order, R_vector_p2);

%% --- Functions---
% window generation function
function [W_Rec, W_Han, W_Ham, n] = generate_windows(N)
% GENERATE_WINDOWS Creates Rectangular, Hanning, and Hamming window vectors.
%
%   [W_Rec, W_Han, W_Ham] = generate_windows(N)
%
%   Inputs:
%       N: Window length (e.g., 1024).
%
%   Outputs:
%       W_Rec: Rectangular window vector.
%       W_Han: Hanning window vector (0.5 - 0.5 * cos(2*pi*n/N)).
%       W_Ham: Hamming window vector (0.54 - 0.46 * cos(2*pi*n/N)).

    % 1. Create time indices (0 to N-1)
    n = 0:N-1;
    
    % 2. Rectangular Window: WRec = 1
    W_Rec = ones(1, N);
    
    % 3. Hanning Window: WHan = 0.5 – 0.5 cos(2πn/N)
    W_Han = 0.5 - 0.5 * cos(2*pi*n/N);
    
    % 4. Hamming Window: WHam = 0.54 – 0.46cos(2πn/N)
    W_Ham = 0.54 - 0.46 * cos(2*pi*n/N);

end
%% ---plot windows in time---
function fig = plot_time_windows(n, W_Rec, W_Han, W_Ham)
% PLOT_TIME_WINDOWS Plots Rectangular, Hanning, and Hamming windows in the time domain.
%
%   fig = plot_time_windows(n, W_Rec, W_Han, W_Ham)
%
%   Inputs:
%       n: Time index vector (e.g., 0:N-1).
%       W_Rec: Rectangular window vector.
%       W_Han: Hanning window vector.
%       W_Ham: Hamming window vector.
%
%   Output:
%       fig: The handle of the created MATLAB figure.

    N = length(n);

    % Create the figure and capture its handle
    fig = figure('Name', 'Window Functions in Time Domain');
    
    % --- X-AXIS LIMIT DEFINITION ---
    X_START = n(1);       % 0
    X_END = n(end);       % N-1 (e.g., 1023)
    
    % Subplot 1: Rectangular Window
    subplot(3, 1, 1);
    plot(n, W_Rec, 'b', 'LineWidth', 1.5);
    title('Rectangular Window ($W_{Rec}$)', 'Interpreter', 'latex');
    ylabel('Amplitude');
    grid on;
    ylim([0 1.1]);
    xlim([X_START X_END]); % <--- Explicitly set X limit
    
    % Subplot 2: Hanning Window
    subplot(3, 1, 2);
    plot(n, W_Han, 'r', 'LineWidth', 1.5);
    title('Hanning Window ($W_{Han}$)', 'Interpreter', 'latex');
    ylabel('Amplitude');
    grid on;
    ylim([0 1.1]);
    xlim([X_START X_END]); % <--- Explicitly set X limit
    
    % Subplot 3: Hamming Window
    subplot(3, 1, 3);
    plot(n, W_Ham, 'g', 'LineWidth', 1.5);
    title('Hamming Window ($W_{Ham}$)', 'Interpreter', 'latex');
    xlabel('Sample Index ($n$)', 'Interpreter', 'latex');
    ylabel('Amplitude');
    grid on;
    ylim([0 1.1]);
    xlim([X_START X_END]); % <--- Explicitly set X limit
    
    % Adjust layout for better visualization
    sgtitle(['Time Domain Representation of Windows (N=', num2str(N), ')'], 'FontSize', 14);
end
%% ---Get Audio Info---
function [Fs, bitDepth, bitRate, numChannels, totalSamples] = get_audio_info(audio_file_path)
% GET_AUDIO_INFO Extracts metadata (Fs, bit depth, bit rate, etc.) from a WAV audio file.
%
%   [Fs, bitDepth, bitRate, numChannels, totalSamples] = get_audio_info(audio_file_path)
%
%   Inputs:
%       audio_file_path: Full path to the audio file (e.g., '..\Data\Test\C02n_1.wav').
%
%   Outputs:
%       Fs: Sampling frequency (Hz).
%       bitDepth: Number of bits per sample.
%       bitRate: Data rate (bits per second).
%       numChannels: Number of audio channels (e.g., 1 for mono, 2 for stereo).
%       totalSamples: Total number of samples in the file.
%

    try
        % Use audioinfo to read the metadata without loading the full audio data
        info = audioinfo(audio_file_path);
        
        % Extract required information
        Fs = info.SampleRate;             % Sampling Frequency (Hz)
        bitDepth = info.BitsPerSample;    % Bits per Sample
        numChannels = info.NumChannels;   % Number of Channels (1=mono, 2=stereo)
        totalSamples = info.TotalSamples; % Total number of samples
        
        % Calculate Bit Rate (Data Rate)
        % Bit Rate = Fs * Bits Per Sample * Number of Channels
        bitRate = Fs * bitDepth * numChannels;
        
        % Display the extracted information
        fprintf('--- Audio File Information ---\n');
        fprintf('File: %s\n', audio_file_path);
        fprintf('Sampling Frequency (Fs): %d Hz\n', Fs);
        fprintf('Bits Per Sample (Bit Depth): %d bits\n', bitDepth);
        fprintf('Number of Channels: %d\n', numChannels);
        fprintf('Total Samples: %d\n', totalSamples);
        fprintf('Calculated Bit Rate: %d bits/s\n', bitRate);
        fprintf('------------------------------\n');
        
    catch ME
        % Handle file not found or other errors gracefully
        fprintf(2, 'Error reading audio file: %s\n', ME.message);
        Fs = []; 
        bitDepth = []; 
        bitRate = []; 
        numChannels = []; 
        totalSamples = []; 
    end
end
%% ---Convert to Frequency Domain ---
function [f, dB_Rec_shifted, dB_Han_shifted, dB_Ham_shifted] = compute_freq_response(Fs, W_Rec, W_Han, W_Ham)
% COMPUTE_FREQ_RESPONSE Calculates the shifted and normalized log-magnitude
%                       frequency response for three window functions.
%
%   [f, dB_Rec_shifted, dB_Han_shifted, dB_Ham_shifted] = compute_freq_response(Fs, W_Rec, W_Han, W_Ham)
%
%   Inputs:
%       Fs: Sampling frequency (Hz).
%       W_Rec, W_Han, W_Ham: Time-domain window vectors (must be the same length N).
%
%   Outputs:
%       f: Frequency vector for plotting (perfectly centered around 0 Hz).
%       dB_Rec_shifted, dB_Han_shifted, dB_Ham_shifted: Shifted log-magnitude
%           spectra (in dB, normalized to 0 dB peak).

    % Get the window length N
    N = length(W_Rec);
    
    %% --- 1. Apply FFT and compute magnitude ---
    fft_Rec = abs(fft(W_Rec));
    fft_Han = abs(fft(W_Han));
    fft_Ham = abs(fft(W_Ham));
    
    %% --- 2. Normalize the magnitude spectrum so the peak is 0 dB ---
    % Normalization is done by dividing by the maximum value before taking log10
    % A small epsilon is added to avoid log(0) which produces -Inf
    epsilon = 1e-15; 
    
    dB_Rec = 20 * log10(fft_Rec / (max(fft_Rec) + epsilon) + epsilon);
    dB_Han = 20 * log10(fft_Han / (max(fft_Han) + epsilon) + epsilon);
    dB_Ham = 20 * log10(fft_Ham / (max(fft_Ham) + epsilon) + epsilon);
    
    %% --- 3. Shift the zero-frequency component to the center (DC at f=0) ---
    dB_Rec_shifted = fftshift(dB_Rec);
    dB_Han_shifted = fftshift(dB_Han);
    dB_Ham_shifted = fftshift(dB_Ham);
    
    %% --- 4. Create the CORRECT frequency vector for the x-axis ---
    % For an even length N (like 1024), the frequency vector should span 
    % from -Fs/2 up to (but excluding) Fs/2.
    f_index = (-N/2 : N/2 - 1);
    f = f_index * (Fs / N);

end
%% ---plot windows in frequency---
function fig = plot_freq_windows(Fs, f, dB_Rec_shifted, dB_Han_shifted, dB_Ham_shifted)
% PLOT_FREQ_WINDOWS Plots the shifted and normalized log-magnitude spectra
%                     of the three window functions in separate subplots.
%
%   fig = plot_freq_windows(Fs, f, dB_Rec_shifted, dB_Han_shifted, dB_Ham_shifted)
%
%   Inputs:
%       Fs: Sampling frequency (Hz).
%       f: Frequency vector.
%       dB_Rec_shifted, dB_Han_shifted, dB_Ham_shifted: Shifted log-magnitude spectra.
%
%   Output:
%       fig: The handle of the created MATLAB figure.

    % Define a magnitude floor (e.g., -80 dB) to clip -Inf and extremely small values for plotting clarity.
    MAGNITUDE_FLOOR = -80; 
    % Zoom limit: +/- Fs/16 (e.g., +/- 1000 Hz for Fs=16000)
    X_LIMIT = Fs / 16; 

    % Apply clipping to the dB values for better visualization
    dB_Rec_clipped = max(dB_Rec_shifted, MAGNITUDE_FLOOR);
    dB_Han_clipped = max(dB_Han_shifted, MAGNITUDE_FLOOR);
    dB_Ham_clipped = max(dB_Ham_shifted, MAGNITUDE_FLOOR);

    % Create the figure and capture its handle
    fig = figure('Name', 'Window Functions in Frequency Domain');
    
    % --- Subplot 1: Rectangular Window ---
    subplot(3, 1, 1);
    plot(f, dB_Rec_clipped, 'b', 'LineWidth', 1.5);
    title('Rectangular Window Frequency Response ($W_{Rec}$)', 'Interpreter', 'latex');
    ylabel('Magnitude (dB)');
    xlim([-X_LIMIT, X_LIMIT]); 
    ylim([MAGNITUDE_FLOOR 0]);      
    grid on;
    
    % --- Subplot 2: Hanning Window ---
    subplot(3, 1, 2);
    plot(f, dB_Han_clipped, 'r', 'LineWidth', 1.5);
    title('Hanning Window Frequency Response ($W_{Han}$)', 'Interpreter', 'latex');
    ylabel('Magnitude (dB)');
    xlim([-X_LIMIT, X_LIMIT]); 
    ylim([MAGNITUDE_FLOOR 0]);      
    grid on;
    
    % --- Subplot 3: Hamming Window ---
    subplot(3, 1, 3);
    plot(f, dB_Ham_clipped, 'g', 'LineWidth', 1.5);
    title('Hamming Window Frequency Response ($W_{Ham}$)', 'Interpreter', 'latex');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    xlim([-X_LIMIT, X_LIMIT]); 
    ylim([MAGNITUDE_FLOOR 0]);      
    grid on;
    
    % Adjust layout for better visualization
    sgtitle(['Frequency Domain Analysis of Windows (N=', num2str(length(f)), ', Fs=', num2str(Fs), ')'], 'FontSize', 14);
end
%% --- LPC_MATRIX_SOLUTION ---
function [a_coeffs, E_error, p_order] = lpc_matrix_solution(R_vector)
% LPC_MATRIX_SOLUTION Solves the Yule-Walker (Normal) equations for LPC coefficients.
%
% This function solves the matrix equation R * a = -r, where R is the 
% autocorrelation matrix, a is the vector of LPC coefficients, and r is 
% the autocorrelation vector.
%
% Inputs:
%   R_vector: The autocorrelation vector, R = [R(0), R(1), R(2), ..., R(p)]
%             where R(0) is the first element.
%
% Outputs:
%   a_coeffs: The LPC coefficients, a = [a1, a2, ..., ap].
%   E_error:  The minimum mean-squared prediction error, E_p.
%   p_order:  The prediction order, p.

% --- 1. Determine Prediction Order (p) ---
% The R_vector must contain p+1 elements: R(0) through R(p).
if length(R_vector) < 2
    error('R_vector must contain at least R(0) and R(1).');
end
p_order = length(R_vector) - 1;

fprintf('Solving LPC system of order p = %d...\n', p_order);

% --- 2. Construct the Autocorrelation Matrix (R) and Vector (r) ---

% The Toeplitz Matrix R (size p x p)
% R is constructed from the first p elements of R_vector.
R_matrix = toeplitz(R_vector(1:p_order), R_vector(1:p_order));

% The right-hand-side vector (-r) (size p x 1)
% r_rhs = [ -R(1); -R(2); ...; -R(p) ]
r_rhs = R_vector(2:end)'; % R_vector(2:end) are R(1) to R(p)

% --- 3. Solve for LPC Coefficients (a) ---
% Solve the linear system: R_matrix * a_coeffs = r_rhs
a_coeffs = R_matrix \ r_rhs;

% --- 4. Calculate the Minimum Mean-Squared Prediction Error (E_p) ---
% E_p = R(0) + sum_{k=1}^{p} a_k * R(k)
% In matrix form: E_p = R(0) + a' * r
E_error = R_vector(1) + a_coeffs' * R_vector(2:end)';

fprintf('LPC Solution Complete.\n');

end
%% --- PRINT R VECTOR ---
function print_R_vector(R_vector)
% PRINT_R_VECTOR prints the autocorrelation vector R in a clean,
% human-readable format, specifying the lag indices.
%
% This function is useful for clearly displaying the input data for LPC
% calculations in DSP assignments or reports.
%
% Inputs:
%   R_vector: The autocorrelation vector, R = [R(0), R(1), R(2), ..., R(p)]
%             where R(0) is the first element.

    % Determine the prediction order 'p'. The length of R_vector is p+1.
    p_order = length(R_vector) - 1;
    
    % --- 1. Construct the R(lag) labels ---
    % Example: R(0), R(1), R(2)
    labels = cell(1, p_order + 1);
    for k = 0:p_order
        labels{k+1} = sprintf('R(%d)', k);
    end
    
    % Combine the labels into a single string: "R(0), R(1), R(2)"
    labels_str = strjoin(labels, ', ');
    
    % --- 2. Construct the R values string (formatted to 4 decimal places) ---
    % Convert the numeric vector to a string, formatted neatly.
    R_values_str = sprintf('%.4f, ', R_vector);
    
    % Remove the trailing comma and space
    R_values_str = R_values_str(1:end-2); 
    
    % --- 3. Print the final output string ---
    % Example: Input Autocorrelation Vector R = [R(0), R(1), R(2)] = [1.0000, 0.7000, 0.4000]
    fprintf('Input Autocorrelation Vector R = [%s] = [%s]\n', labels_str, R_values_str);

end
%% --- LPC results  ---
function print_lpc_results(a_coeffs, E_error, p_order, R_vector)
% PRINT_LPC_RESULTS prints the calculated LPC coefficients and prediction error.
%
% This function formats and displays the results of the Yule-Walker
% matrix solution in a clean, human-readable format.
%
% Inputs:
%   a_coeffs: The vector of LPC coefficients, [a1, a2, ..., ap].
%   E_error:  The minimum mean-squared prediction error, E_p.
%   p_order:  The prediction order, p.
%   R_vector: The autocorrelation vector, R = [R(0), R(1), R(2), ..., R(p)]
    
    % --- Call the new printing function to display the input cleanly ---
    print_R_vector(R_vector);

    % prediction prder
    fprintf('\n--- LPC Results (p=%d) ---\n', p_order);
    fprintf('Prediction Order (p): %d\n', p_order);

    
    
    % Loop through the coefficient vector to print each coefficient individually
    num_coeffs = length(a_coeffs);
    for k = 1:num_coeffs
        fprintf('LPC Coefficient a%d: %.4f\n', k, a_coeffs(k));
    end
    
    % Print the minimum prediction error
    fprintf('Minimum Mean-Squared Error (E_%d): %.4f\n', p_order, E_error);
    fprintf('------------------------\n');

end
%% ---save figure to picture---
function figure_to_png(figHandle, filename, relative_save_path)
% FIGURE_TO_PNG Saves a specified MATLAB figure handle as a high-quality PNG file.
%
%   figure_to_png(figHandle, filename, relative_save_path)
%
%   Inputs:
%       figHandle: Handle to the figure object.
%       filename: The base filename (e.g., 'problem1_time_domain').
%       relative_save_path: The desired save path relative to the *current* %                           function's directory (e.g., '../Data/Results/plot').
%
%   Output:
%       Saves the figure as a PNG file in the specified directory structure.

    % 1. Determine the current directory of this function (e.g., .../DSP_Speech_Processing/Src)
    current_file_path = mfilename('fullpath');
    current_dir = fileparts(current_file_path);

    % 2. Construct the absolute output directory path by combining the current directory 
    %    with the relative path provided by the user.
    %    fullfile handles platform-specific path separators (\ or /).
    outputDir = fullfile(current_dir, relative_save_path);
    
    % 3. Ensure the output directory exists
    if ~exist(outputDir, 'dir')
        % Try to create the directory and all parent directories if necessary
        [success, message, ~] = mkdir(outputDir);
        if ~success
            fprintf(2, 'Error creating directory %s: %s\n', outputDir, message);
            return; % Exit the function if directory creation fails
        end
    end
    
    % 4. Construct the full file path for saving
    fullPath = fullfile(outputDir, [filename, '.png']);
    
    try
        % Set common properties for high-quality export
        set(figHandle, 'PaperPositionMode', 'auto');
        set(figHandle, 'Renderer', 'painters'); 
        
        % Save the figure to PNG format (300 DPI resolution)
        print(figHandle, fullPath, '-dpng', '-r300');
        
        fprintf('Successfully saved figure to: %s\n', fullPath);
        
    catch ME
        % Display an error message if saving fails
        fprintf(2, 'Error saving figure %s: %s\n', filename, ME.message);
        disp('Check if the figure handle is valid or if file permissions allow saving.');
    end
end