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

%% Problem2 LPC Analysis based on given R data
% --- 1. Define Input Data for Problem 2 ---

% R_vector = [R(0), R(1), R(2), ..., R(p)]
R_matrix2 = [1, 0.7, 0.4];
LPC_ORDER_p2 = length(R_matrix2) - 1; % Should be 2
frames=[1,2,3];

% print it
print_R_matrix(R_matrix2);

fprintf('--- Problem 2: LPC (p=%d) Solution Verification ---\n',LPC_ORDER_p2);


% --- 5. Call the generalized solver function  ---
[A_matrix2, E_vector2,P2] = lpc_matrix_solution(R_matrix2);

% --- 7. Display Results ---
problem2_fig_handle = print_lpc_matrix_results(frames, A_matrix2, E_vector2, R_matrix2);

%  Save the frame grid figure
figure_to_png(problem2_fig_handle, 'problem2_sol',relative_path_to_plots); 


%% Problem 3 Frame Analysis based on given S data
% Problem: s[n] = [1, 4, 0, -4, -1, 2, 4, -1, 2, 5], Frame Size = 6, Overlap = 2, LPC Order P = 2.
% --- 1. Define Input Data for Problem 3 ---
% Signal (s0 to s9)
s = [1, 4, 0, -4, -1, 2, 4, -1, 2, 5];

frame_length = 6; % Frame Size
P = 2; % LPC Order
overlap = 2;
frame_shift = frame_length - overlap;

fprintf('--- Problem 3a: LPC Analysis for Frame Size N=%d and Order P=%d ---\n', frame_length, P);


% --- 2. Call the generalized solver function  ---
frames3a = extract_frames(s, frame_length, frame_shift);

% --- 3. Display Results ---
% Display the full signal with markers 
frame_fig_handle = plot_frames(s, frames3a, frame_length, frame_shift, Fs);

%  Save the frame figure
figure_to_png(frame_fig_handle, 'problem3a_frames',relative_path_to_plots); 

% Display individual frames in a grid 
frame_grid_handle = plot_frame_grid(frames3a, Fs);

%  Save the frame grid figure
figure_to_png(frame_grid_handle, 'problem3a_frames_gird',relative_path_to_plots); 

% --- 4. calculate autocorrelation  ---
R_matrix3a = calculate_autocorr_frames(frames3a, P);

% print it
print_R_matrix(R_matrix3a);

fprintf('--- Problem 3a: LPC (p=%d) Solution Verification ---\n',P);

% --- 5. Call the generalized solver function  ---
[A_matrix3a, E_vector3a,P3a] = lpc_matrix_solution(R_matrix3a);

% --- 7. Display Results ---
problem3a_fig_handle = print_lpc_matrix_results(frames3a, A_matrix3a, E_vector3a, R_matrix3a);

%  Save the frame grid figure
figure_to_png(problem3a_fig_handle, 'problem3a_sol',relative_path_to_plots); 

%% Problem 3b: LPC Analysis with Pre-emphasis (alpha=0.96)
% --- 1. Define Input Data for Problem 3 ---
% Signal: s[n] = [1, 4, 0, -4, -1, 2, 4, -1, 2, 5], Frame Size = 6, Overlap = 2, LPC Order P = 2.
% Pre-emphasis constant: alpha = 0.96
s = [1, 4, 0, -4, -1, 2, 4, -1, 2, 5];

frame_length = 6; % Frame Size
P = 2; % LPC Order
overlap = 2;
frame_shift = frame_length - overlap;
pre_alpha = 0.96;

% apply pre emmphasis
s_emph = pre_emphasis_signal(s, pre_alpha);

fprintf('--- Problem 3b: LPC Analysis with Pre-emphasis constant: alpha = %.2f  for Frame Size N=%d and Order P=%d ---\n',pre_alpha, frame_length, P);

% --- 2. Call the generalized solver function  ---
frames3b = extract_frames(s_emph, frame_length, frame_shift);

% --- 3. Display Results ---
% Display the full signal with markers 
frame_fig_handle = plot_frames(s_emph, frames3b, frame_length, frame_shift, Fs);

%  Save the frame figure
figure_to_png(frame_fig_handle, 'problem3b_frames',relative_path_to_plots); 

% Display individual frames in a grid 
frame_grid_handle = plot_frame_grid(frames3b, Fs);

%  Save the frame grid figure
figure_to_png(frame_grid_handle, 'problem3b_frames_gird',relative_path_to_plots); 

% --- 4. calculate autocorrelation  ---
R_matrix3b = calculate_autocorr_frames(frames3b, P);

% print it
print_R_matrix(R_matrix3b);

fprintf('--- Problem 3b: LPC (p=%d) Solution Verification ---\n',P);

% --- 5. Call the generalized solver function  ---
[A_matrix3b, E_vector3b,P3b] = lpc_matrix_solution(R_matrix3b);

% --- 7. Display Results ---
problem3b_fig_handle = print_lpc_matrix_results(frames3b, A_matrix3b, E_vector3b, R_matrix3b);

%  Save the frame grid figure
figure_to_png(problem3b_fig_handle, 'problem3b_sol',relative_path_to_plots); 


%% Problem 4: Formant and Bandwidth Estimation from All-Pole System
% An 8th-order all-pole system modeled by 4 complex-conjugate pole pairs.
% Given poles (4 complex-conjugate pairs)
% The system function is H(z) = 1 / A(z), where A(z) is the polynomial whose roots are these poles.
Poles = [
    0.965500 + 0.192050i; 0.965500 - 0.192050i; % p1,2
    0.812108 + 0.542633i; 0.812108 - 0.542633i; % p3,4
    0.534176 + 0.799451i; 0.534176 - 0.799451i; % p5,6
    0.183930 + 0.924681i; 0.183930 - 0.924681i  % p7,8
];
% --- Part a: Calculate and Plot the magnitude spectrum in dB  ---

% 1. Call the function to calculate A_coeffs and H_mag
[A_coeffs, H_mag, W] = calculate_all_pole_response(Poles, Fs, N);

% --- Part a: Plot the magnitude spectrum in dB ---
problem4_fig_handle = plot_pole_spectrum(Fs, W, H_mag);


% --- Part b: Estimate formants and bandwidths ---
[Formants_Hz, Bandwidths_Hz] = estimate_and_plot_formants(Poles, Fs, H_mag);

%  Save the magnitude spectrum figure
figure_to_png(problem4_fig_handle, 'problem4_plot',relative_path_to_plots);

plot_BW_poles(Poles, Fs);
%}
%%  Problem 5: Recover Poles from A(z)
A_coeffs_verify = [1.0, -4.9914283, 12.3717836, -19.81615903, 22.40030463,...
    -18.3112730, 10.60283765, -3.99936958, 0.75965617]
Recovered_Poles = recover_poles_from_coeffs(A_coeffs_verify);

% 1. Call the function to verify A_coeffs and H_mag
[Recovered_A_coeffs, Recovered_H_mag, Recovered_W] = calculate_all_pole_response(Recovered_Poles, Fs, N);

% --- Part a: Plot the magnitude spectrum in dB ---
problem4_fig_handle = plot_pole_spectrum(Fs, Recovered_W, Recovered_H_mag);

% --- Estimate formants and bandwidths ---
[Recovered_Formants_Hz, Recovered_Bandwidths_Hz] = estimate_and_plot_formants(Recovered_Poles, Fs, Recovered_H_mag);

%  Save the magnitude spectrum figure
figure_to_png(problem4_fig_handle, 'problem5_plot',relative_path_to_plots);

fig_handle =plot_BW_poles(Recovered_Poles, Fs);


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
    fft_Rec = abs(fft(W_Rec,N));
    fft_Han = abs(fft(W_Han,N));
    fft_Ham = abs(fft(W_Ham,N));
   

    
    %% --- 2. Normalize the magnitude spectrum so the peak is 0 dB ---
    % Normalization is done by dividing by the maximum value before taking log10
    % A small epsilon is added to avoid log(0) which produces -Inf
    epsilon = 1e-12; 
    
    dB_Rec = 20 * log10(fft_Rec + epsilon);
    dB_Han = 20 * log10(fft_Han);
    dB_Ham = 20 * log10(fft_Ham);

    
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
    
    MAGNITUDE_FLOOR = -400; 
    % Zoom limit: +/- Fs/16 (e.g., +/- 1000 Hz for Fs=16000)
    X_LIMIT = Fs/10;

    % Apply clipping to the dB values for better visualization
    dB_Rec_shifted = max(dB_Rec_shifted, MAGNITUDE_FLOOR);
    dB_Han_shifted = max(dB_Han_shifted, MAGNITUDE_FLOOR);
    dB_Ham_shifted = max(dB_Ham_shifted, MAGNITUDE_FLOOR);

    % Create the figure and capture its handle
    fig = figure('Name', 'Window Functions in Frequency Domain');
    
    % --- Subplot 1: Rectangular Window ---
    subplot(3, 1, 1);
    
    plot(f, dB_Rec_shifted, 'b');
    title('Rectangular Window Frequency Response ($W_{Rec}$)', 'Interpreter', 'latex');
    ylabel('Magnitude (dB)');
    xlim([-X_LIMIT, X_LIMIT]);  
    grid on;
    
    % --- Subplot 2: Hanning Window ---
    subplot(3, 1, 2);
    plot(f, dB_Han_shifted, 'r');
    title('Hanning Window Frequency Response ($W_{Han}$)', 'Interpreter', 'latex');
    ylabel('Magnitude (dB)');
    xlim([-X_LIMIT, X_LIMIT]);  
    grid on;
    
    % --- Subplot 3: Hamming Window ---
    subplot(3, 1, 3);
    plot(f, dB_Ham_shifted, 'g');
    title('Hamming Window Frequency Response ($W_{Ham}$)', 'Interpreter', 'latex');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    xlim([-X_LIMIT, X_LIMIT]);  
    grid on;
    
    % Adjust layout for better visualization
    sgtitle(['Frequency Domain Analysis of Windows (N=', num2str(length(f)), ', Fs=', num2str(Fs), ')'], 'FontSize', 14);
end
%% --- LPC_MATRIX_SOLUTION ---
function [A_matrix, E_vector,P] = lpc_matrix_solution(R_matrix)
%LPC_MATRIX_SOLUTION Solves the Yule-Walker (Normal) equations for LPC coefficients 
%   for multiple frames using direct matrix inversion.
%
%   [A_MATRIX, E_VECTOR] = lpc_matrix_solution(R_MATRIX)
%
% Inputs:
%   R_matrix : Matrix where each ROW contains the autocorrelation 
%              coefficients for one frame: [R[0], R[1], ..., R[P]].
%
% Outputs:
%   A_matrix : Matrix where each ROW contains the final filter denominator 
%              coefficients: [ a[1], a[2], ..., a[P]].
%   E_vector : Column vector containing the minimum mean-squared error (E_P) 
%              for each frame.

    % --- 1. Setup and Pre-allocation ---
    [num_frames, P_plus_1] = size(R_matrix);
    P = P_plus_1 - 1; % Prediction order P
    
    % A_matrix will store the final filter denominator coefficients A = [ a1, ...]
    A_matrix = zeros(num_frames, P);
    
    % E_vector stores the final prediction error E_P
    E_vector = zeros(num_frames, 1);
    
    fprintf('Solving LPC system of order p = %d for %d frames via Matrix Inversion...\n', P, num_frames);

    % --- 2. Process Each Frame ---
    for k = 1:num_frames
        R_vector = R_matrix(k, :); % R_vector = [R(0), R(1), ..., R(P)]
        
        % R_0 is R(0)
        R_0 = R_vector(1); 
        
        % Check for zero energy frame
        if R_0 < 1e-10
            A_matrix(k, :) = [1, zeros(1, P)]; 
            E_vector(k) = 0;
            continue; 
        end
        
        % --- 3. Construct the Autocorrelation Matrix (R) and Vector (-r) ---
        
        % The Toeplitz Matrix R_P (size P x P) from R(0) to R(P-1)
        % R_vector(1:P) are R(0) to R(P-1)
        R_P_matrix = toeplitz(R_vector(1:P), R_vector(1:P));
        
        % The right-hand-side vector (-r_P) (size P x 1)
        % r_P = [R(1); R(2); ...; R(P)]. The RHS is -r_P.
        % R_vector(2:end) are R(1) to R(P)
        r_P_rhs = R_vector(2:end)';
        
        % --- 4. Solve for Predictor Coefficients (a) ---
        % Solve the linear system: R_P_matrix * a_coeffs = -r_P
        % a_coeffs = [a1, a2, ..., aP]
        a_predictor_coeffs = R_P_matrix \ r_P_rhs;
        
        % --- 5. Calculate the Minimum Mean-Squared Prediction Error (E_P) ---
        % E_P = R(0) + sum_{k=1}^{P} a_k * R(k) = R(0) + a' * r_P
        % Note: R_vector(2:end)' is the vector r_P
        E_P_error = R_0 + a_predictor_coeffs' * R_vector(2:end)';
        
        % --- 6. Store Final Results ---
        % The filter denominator coefficients are A = [ a1, a2, ..., aP]
        A_matrix(k, :) = [ a_predictor_coeffs'];
        E_vector(k) = E_P_error;
    end
    
    fprintf('Matrix Solution Complete. (Filter A(z) coefficients returned: [a1, ..., aP])\n');
end
%% --- PRINT R Matrix ---
function print_R_matrix(R_matrix)
% PRINT_R_VECTOR prints the autocorrelation matrix R in a clean,
% human-readable format, specifying the lag indices for each frame.
%
% This function is updated to handle a matrix input (multiple frames).
%
% Inputs:
%   R_matrix: The autocorrelation matrix, where each ROW is a frame's
%             autocorrelation vector R = [R(0), R(1), ..., R(p)]
    
    % Get dimensions of the R matrix
    [num_frames, P_plus_1] = size(R_matrix);
    % Determine the prediction order 'p'
    P_order = P_plus_1 - 1;
    
    % --- 1. Construct the R(lag) labels (only needs to be done once) ---
    labels = cell(1, P_plus_1);
    for k = 0:P_order
        labels{k+1} = sprintf('R(%d)', k);
    end
    labels_str = strjoin(labels, ', ');
    
    fprintf('\n--- Autocorrelation Analysis: R[0] to R[%d] for %d Frames ---\n', P_order, num_frames);
    
    % --- 2. Loop through each frame (row of the matrix) and print ---
    for frame_idx = 1:num_frames
        % Extract the R vector for the current frame
        R_vector = R_matrix(frame_idx, :);
        
        % Construct the R values string (formatted to 4 decimal places)
        R_values_str = sprintf('%.4f, ', R_vector);
        
        % Remove the trailing comma and space
        R_values_str = R_values_str(1:end-2); 
        
        % Print the final output string for the current frame
        fprintf('Frame %d R = [%s] = [%s]\n', frame_idx, labels_str, R_values_str);
    end
    fprintf('------------------------------------------------------------------\n');
end
%% --- LPC results  ---
function figHandle = print_lpc_matrix_results(X_frames, A_matrix, E_vector, R_matrix)
%PRINT_LPC_MATRIX_RESULTS Generates a MATLAB figure containing a formatted 
%   table of LPC analysis results. This version formats all numeric data
%   to strings to ensure compatibility with various MATLAB versions.
%
% Inputs:
%   X_frames: Matrix where each ROW contains the raw signal samples for the frame.
%   A_matrix: Matrix where each ROW contains the LPC predictor 
%             coefficients: [a[1], a[2], ..., a[P]].
%   E_vector: Column vector containing the minimum mean-squared error (E_P) 
%             for each frame.
%   R_matrix: Matrix where each ROW contains the autocorrelation 
%             coefficients: [R[0], R[1], ..., R[P]].
%
% Output:
%   figHandle: Handle to the generated figure.

    % --- 1. Determine Dimensions and Order ---
    [num_frames, P] = size(A_matrix);
    
    if num_frames == 0
        figHandle = [];
        fprintf('--- LPC Results ---\n');
        fprintf('No frames processed. A_matrix is empty.\n');
        return;
    end
    
    [~, num_samples_per_frame] = size(X_frames);
    MAX_SAMPLES_TO_SHOW = 6; % Show up to 6 samples
    samples_to_print = min(num_samples_per_frame, MAX_SAMPLES_TO_SHOW);
    
    % --- 2. Create Headers ---
    headers = {'Frame Index'};
    
    % Add Samples Header
    sample_header_str = sprintf('Samples (first %d)', samples_to_print);
    if num_samples_per_frame > MAX_SAMPLES_TO_SHOW
        sample_header_str = [sample_header_str, ' ...'];
    end
    headers = [headers, sample_header_str];

    % Add R Coefficients Headers
    for p_idx = 0:P
        headers = [headers, sprintf('R(%d)', p_idx)];
    end
    
    % Add Error Header
    headers = [headers, 'Error (E_P)'];
    
    % Add A Coefficients Headers
    for p_idx = 1:P
        headers = [headers, sprintf('a%d', p_idx)];
    end
    
    % --- 3. Format Data into Cell Array of Strings ---
    % We convert all numerical data to strings with 4 decimal places for display.
    data_cell = cell(num_frames, length(headers));
    
    for k = 1:num_frames
        
        % 1. Frame Index (Formatted to String)
        data_cell{k, 1} = sprintf('%d', k);
        
        % 2. Samples String (Formatted for display)
        sample_str = '';
        for i = 1:samples_to_print
            % Use %s.0f for integer-like samples
            sample_str = [sample_str, sprintf('%.2f ', X_frames(k, i))];
        end
        if num_samples_per_frame > MAX_SAMPLES_TO_SHOW
            sample_str = [sample_str, '...'];
        end
        data_cell{k, 2} = strtrim(sample_str); 
        
        col_idx = 3; 

        % 3. R Coefficients (Formatted to String)
        for r_val = R_matrix(k, :)
            data_cell{k, col_idx} = sprintf('%.4f', r_val);
            col_idx = col_idx + 1;
        end
        
        % 4. Error (E_P) (Formatted to String)
        data_cell{k, col_idx} = sprintf('%.4f', E_vector(k));
        col_idx = col_idx + 1;
        
        % 5. A Coefficients (Formatted to String)
        for a_val = A_matrix(k, :)
            data_cell{k, col_idx} = sprintf('%.4f', a_val);
            col_idx = col_idx + 1;
        end
    end
    
    % --- 4. Create Figure and Uitable ---
    figHandle = figure('Name', 'LPC Analysis Results Table', 'NumberTitle', 'off', 'Color', 'w');
    
    u = uitable(figHandle);
    u.Data = data_cell;
    u.ColumnName = headers;
    u.RowName = {}; % Hide row numbers
    u.Units = 'normalized';
    u.Position = [0.02 0.02 0.96 0.93]; % Full figure size with margin
    u.FontSize = 10;
    
    % Since all data is now stored as strings, we don't need 'ColumnFormat' 
    % or 'ColumnDisplayFormat', increasing compatibility.
    
    % Set column widths for better fit
    col_widths = {50, 'auto'}; % Frame index width and Samples auto-width
    fixed_width = {70}; 
    for i = 3:length(headers)
        col_widths = [col_widths, fixed_width];
    end
    u.ColumnWidth = col_widths;

    % Add title to the figure
    title_str = sprintf('LPC Analysis Results Table (Prediction Order P=%d)', P);
    uicontrol('Style', 'text', 'String', title_str, ...
              'Units', 'normalized', 'Position', [0.3 0.95 0.4 0.05], ...
              'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'w');

    % Maximize the figure for better view if it has many columns
    set(figHandle, 'WindowState', 'maximized');
    
    fprintf('\nLPC Results Table successfully generated in a figure window (handle: %d).\n', figHandle.Number);
end
%% --- Get Frames  ---
function frames = extract_frames(signal, frame_length, frame_shift)
%EXTRACT_FRAMES Divides a 1D signal (e.g., speech) into frames, with zero-padding.
%
%   FRAMES = extract_frames(SIGNAL, FRAME_LENGTH, FRAME_SHIFT) divides the
%   input SIGNAL vector into a matrix of frames. Each ROW of the output
%   matrix FRAMES represents one frame. The last frame is zero-padded to
%   ensure the entire original signal is covered.
%
% Inputs:
%   signal       : The input signal vector (e.g., audio data).
%   frame_length : The length of each frame, in samples (N).
%   frame_shift  : The step size between the start of successive frames (the hop size), in samples (S).
%                  Overlap is calculated as N - S.
%
% Output:
%   frames       : A matrix where each ROW is a single frame of the signal.
%                  Size is [Num_Frames, FRAME_LENGTH].
%
% Note: This implementation uses zero-padding to ensure the entire original
%       signal is covered by the frames.

% --- 1. Input Validation and Setup ---
signal = signal(:); % Ensure the input signal is a column vector
signal_length = length(signal);

% Ensure parameters are positive integers and shift is <= length
if frame_length <= 0 || frame_shift <= 0 || frame_shift > frame_length
    error('Frame length (N) and shift (S) must be positive integers, and S must be <= N.');
end

% --- 2. Calculate Framing Parameters and Zero-Padding ---

% Calculate the total number of frames required (K).
% K = 1 + ceil((L - N) / S), where L=signal_length, N=frame_length, S=frame_shift
if signal_length < frame_length
    % If the signal is shorter than one frame, we still need one padded frame.
    num_frames = 1;
else
    % Standard calculation for zero-padded framing
    num_frames = 1 + ceil((signal_length - frame_length) / frame_shift);
end

% Calculate the total signal length required after padding (L_padded).
% L_padded = (K - 1) * S + N
required_length = (num_frames - 1) * frame_shift + frame_length;

% Calculate how many zeros to append
padding_amount = required_length - signal_length;

% Apply padding only if necessary
if padding_amount > 0
    signal = [signal; zeros(padding_amount, 1)];
end

% --- 3. Frame Extraction (as Columns first, for efficient memory access) ---

% Pre-allocate the output matrix with frames as columns: [N x K]
temp_frames_as_cols = zeros(frame_length, num_frames);

% Loop through and extract each frame
for k = 1:num_frames
    % Calculate the starting index for the current frame
    start_index = 1 + (k - 1) * frame_shift;
    
    % The ending index is calculated relative to the start index
    end_index = start_index + frame_length - 1;
    
    % Extract the segment and store it as a column in the temporary matrix
    temp_frames_as_cols(:, k) = signal(start_index:end_index);
end

% --- 4. Transpose to finalize output: Frames as Rows ---
% Transpose the matrix from [N x K] to [K x N]
frames = temp_frames_as_cols.';

end
%% --- Plot Frames  ---
function fig_handle = plot_frames(s, frames, frame_length, frame_shift, Fs)
%PLOT_FRAMES Plots the original signal and visualizes individual frames using subplots.
%
%   FIG_HANDLE = plot_frames(S, FRAMES, FRAME_LENGTH, FRAME_SHIFT, Fs)
%   Generates a figure with two subplots:
%   1. Top Subplot: Time-domain plot of the entire signal (S) with markers
%      showing the start of each frame.
%   2. Bottom Subplot: Overlays the first 5 extracted frames to visualize 
%      the content and overlap.
%
% Inputs:
%   s            : The original input signal vector.
%   frames       : The frame matrix, where each ROW is a frame (size [K x N]).
%   frame_length : The length of each frame, in samples (N).
%   frame_shift  : The hop size (step size) between frames, in samples (S).
%   Fs           : The sampling frequency, in Hz (used for time scaling).
%
% Output:
%   fig_handle   : Handle to the generated figure.

    % Ensure input signal is a column vector
    s = s(:);
    signal_length = length(s);
    num_frames = size(frames, 1); 
    
    % --- Time Axis Setup ---
    time_s = (0:signal_length-1) / Fs; % Time vector for the original signal in seconds
    start_indices = 1 + (0:(num_frames-1)) * frame_shift;
    start_time = (start_indices - 1) / Fs;
    
    % Get the max amplitude of the signal for consistent scaling
    max_amp = max(abs(s)) * 1.05;

    % --- 1. Create Figure ---
    fig_handle = figure;
    set(fig_handle, 'Color', 'w'); % Set background to white

    % --------------------------------------------------------------------
    % Subplot 1: Full Signal Visualization
    % --------------------------------------------------------------------
    subplot(2, 1, 1);
    plot(time_s, s, 'k', 'LineWidth', 1.5);
    hold on;
    
    % Plot vertical lines at the start of each frame
    for i = 1:num_frames
        t = start_time(i);
        
        % Check for zero-padded frame (last frame)
        is_padded = start_indices(i) + frame_length - 1 > signal_length;
        
        if is_padded
            % Last zero-padded frame (red, dotted)
            plot([t, t], [-max_amp, max_amp], 'r:', 'LineWidth', 0.8, 'HandleVisibility', 'off');
        else
            % Normal frames (blue, dashed)
            plot([t, t], [-max_amp, max_amp], 'b--', 'LineWidth', 1, 'HandleVisibility', 'off');
        end
        
        % Label the frame start
        if i <= 5 || i == num_frames
             text(t, max_amp, sprintf('F%d', i), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'm');
        end
    end

    % Styling
    title('1. Full Signal with Frame Start Markers', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Time (s)', 'FontSize', 10);
    ylabel('Amplitude', 'FontSize', 10);
    grid on;
    xlim([0, time_s(end)]); 
    ylim([-max_amp, max_amp]);
    hold off;

    % --------------------------------------------------------------------
    % Subplot 2: First Few Frames Overlaid
    % --------------------------------------------------------------------
    subplot(2, 1, 2);
    
    % Determine how many frames to plot in the detail view
    frames_to_plot = min(5, num_frames); 
    
    if frames_to_plot > 0
        % Time vector for a single frame (relative time from 0 to N-1 samples)
        time_frame = (0:frame_length-1) / Fs;
        
        % Define colors for different frames
        colors = lines(frames_to_plot); 
        
        for i = 1:frames_to_plot
            % Plot the frame data. Frames are rows, so frames(i, :)
            plot(time_frame, frames(i, :), 'Color', colors(i, :), 'LineWidth', 1, 'DisplayName', sprintf('Frame %d', i));
            hold on;
        end
        
        % Styling
        title(sprintf('2. Detail View: First %d Extracted Frames Overlaid', frames_to_plot), 'FontSize', 12, 'FontWeight', 'bold');
        xlabel('Time within Frame (s)', 'FontSize', 10);
        ylabel('Amplitude', 'FontSize', 10);
        legend('Location', 'southeast');
        grid on;
        xlim([0, time_frame(end)]);
        ylim([-max_amp, max_amp]);
    else
        % Handle case where signal is too short
        text(0.5, 0.5, 'No frames extracted (Signal too short).', 'HorizontalAlignment', 'center');
    end

    hold off;
end
%% --- Plot Frames grid  ---
function fig_handle = plot_frame_grid(frames, Fs)
%PLOT_FRAME_GRID Plots the first N frames in a grid of subplots.
%
%   FIG_HANDLE = plot_frame_grid(FRAMES, Fs)
%   Generates a figure showing the waveform of the first 12 extracted 
%   frames, each in its own subplot, arranged in a 3x4 grid. This is useful 
%   for inspecting the exact content and overlap of the first few frames.
%
% Inputs:
%   frames       : The frame matrix, where each ROW is a frame.
%   Fs           : The sampling frequency, in Hz.
%
% Output:
%   fig_handle   : Handle to the generated figure.

    % --- Configuration ---
    max_frames_to_show = 12;
    rows = 3;
    cols = 4;
    
    num_frames = size(frames, 1);
    
    if num_frames == 0
        warning('PLOT_FRAME_GRID: No frames available to plot (Signal too short).');
        return;
    end

    % Determine the number of frames to actually plot
    frames_to_plot = min(max_frames_to_show, num_frames); 
    
    % --- Create Figure ---
    fig_handle = figure;
    % Make the figure large for better readability of the subplots
    set(fig_handle, 'Color', 'w', 'Units', 'normalized', 'OuterPosition', [0.1 0.1 0.8 0.8]); 
    
    % Get max amplitude across all selected frames for consistent Y-axis scaling
    max_amp = max(abs(frames(1:frames_to_plot, :)), [], 'all') * 1.05;

    % Time vector for a single frame (relative time from 0 to N-1 samples)
    frame_length = size(frames, 2);
    time_frame = (0:frame_length-1) / Fs;

    % Loop through the frames and generate subplots
    for i = 1:frames_to_plot
        % Select the subplot position
        subplot(rows, cols, i);
        
        % Plot the individual frame
        plot(time_frame, frames(i, :), 'b', 'LineWidth', 1.2);
        
        % Styling
        title(sprintf('Frame %d', i), 'FontSize', 10, 'FontWeight', 'bold');
        xlabel('Time (s)', 'FontSize', 8);
        ylabel('Amplitude', 'FontSize', 8);
        
        % Set consistent limits for all frames
        xlim([0, time_frame(end)]);
        ylim([-max_amp, max_amp]);
        grid on;
    end
    
    % Super title for the entire figure
    sgtitle(sprintf('Detailed Visualization: First %d Individual Speech Frames (%d samples/frame)', frames_to_plot, frame_length), ...
            'FontSize', 14, 'FontWeight', 'bold');

end
%% --- calculate autocorrelation  ---
function R_matrix = calculate_autocorr_frames(frames, P)
%CALCULATE_AUTOCORR_FRAMES Calculates the autocorrelation coefficients for each frame.
%
%   R_MATRIX = calculate_autocorr_frames(FRAMES, P) computes the first P+1
%   autocorrelation coefficients (R[0] to R[P]) for every frame in the input
%   matrix FRAMES.
%
% Inputs:
%   frames : Matrix where each ROW is a single signal frame (K x N).
%   P      : The order of the Linear Predictive Coding (LPC order). The 
%            output R_matrix will have P+1 columns (R[0] to R[P]).
%
% Output:
%   R_matrix : A matrix where each ROW contains the autocorrelation 
%              coefficients for one frame. Size is [K x (P+1)].

    % --- 1. Setup and Pre-allocation ---
    num_frames = size(frames, 1);
    frame_length = size(frames, 2);
    
    % The output matrix will have K rows (frames) and P+1 columns (R[0]..R[P])
    R_matrix = zeros(num_frames, P + 1);
    
    % --- 2. Iterative Autocorrelation Calculation ---
    % Loop through each frame (row of the input matrix)
    for k = 1:num_frames
        % Get the current frame (ensure it's treated as a row vector)
        current_frame = frames(k, :);
        
        % Calculate the full autocorrelation sequence R_full using xcorr
        % R_full will have 2*N - 1 elements.
        R_full = xcorr(current_frame);
        
        % The lag 0 correlation (R[0]) is located at the center of R_full.
        % For a sequence of length N, xcorr gives 2N-1 values.
        % The center index (lag 0) is N.
        lag0_index = frame_length;
        
        % Extract the positive lags from R[0] up to R[P].
        % This corresponds to indices: R[lag0_index] to R[lag0_index + P]
        autocorr_coeffs = R_full(lag0_index : lag0_index + P);
        
        % Store the resulting P+1 coefficients in the output matrix
        R_matrix(k, :) = autocorr_coeffs;
    end
    
end
%% --- Pre-empahsis signal  ---
function s_emph = pre_emphasis_signal(s, alpha)
%PRE_EMPHASIS_FILTER_SIGNAL Applies a first-order pre-emphasis filter to a signal.
%
%   s_emph = pre_emphasis_filter_signal(s, alpha) applies the pre-emphasis
%   filter defined by H(z) = 1 - alpha*z^-1 to the input signal s.
%   This is equivalent to the time-domain equation:
%   s_emph[n] = s[n] - alpha * s[n-1]
%
% Inputs:
%   s     - The input time-domain signal (vector).
%   alpha - The pre-emphasis constant (e.g., 0.96 for speech analysis).
%
% Output:
%   s_emph - The pre-emphasized output signal (vector).
%
% The first sample of the output signal is s_emph[1] = s[1] - alpha*s[0].
% MATLAB's 'filter' function correctly handles the implied initial condition s[0]=0.

    % Define the coefficients for the FIR filter H(z) = B(z) / A(z)
    % B(z) is the numerator polynomial: B = [1, -alpha]
    % A(z) is the denominator polynomial: A = [1]
    B = [1, -alpha];
    A = 1;

    % Apply the filter using MATLAB's 'filter' function.
    s_emph = filter(B, A, s);

    %results 
    fprintf('\n--- Problem 3b: LPC Analysis w/ Pre-emphasis (alpha=%.2f) ---\n', alpha);
    fprintf('Original Signal s[n]: [%s]\n', num2str(s));
    fprintf('Pre-emphasized s_emph[n]: [%s]\n', num2str(s_emph, 4));
end
%% --- Calculate Pole Response ---
function [A_coeffs, H_mag, W] = calculate_all_pole_response(Poles, Fs, N_points)
% CALCULATE_ALL_POLE_RESPONSE Calculates the coefficients and magnitude 
% spectrum of an all-pole system H(z) = 1/A(z).
%
%   Inputs:
%       Poles: Array of the system's poles.
%       Fs: Sampling frequency in Hz.
%
%   Outputs:
%       A_coeffs: Denominator coefficients A(z) = 1 + a1*z^-1 + ...
%       H_mag: The magnitude of the frequency response H(e^j\omega).

    % 1. Calculate the denominator coefficients A(z) = 1 + a1*z^-1 + ...
    % The 'poly' function takes the roots (poles) and returns the 
    % coefficients of the polynomial.
    A_coeffs = poly(Poles);
    
    % Create the frequency vector (w) for plotting 
    W = (0:N_points-1) * (Fs/2) / N_points; 
    
    % freqz(Numerator, Denominator, N_points, Sampling_Frequency)
    [H, ~] = freqz(1, A_coeffs, N_points, Fs); 
    
    % Store the magnitude (abs(H))
    H_mag = abs(H);

end
%% --- Poles Plot
function fig_handle =plot_pole_spectrum(Fs, w, H_mag)
% PLOT_POLE_SPECTRUM Plots the magnitude spectrum of an all-pole system in dB.
%
%   Inputs:
%       Fs: Sampling frequency in Hz.
%       w: Frequency vector in Hz.
%       H_mag: Magnitude of the frequency response H(e^j\omega).

    % Plotting the magnitude in dB
    fig_handle= figure('Name', 'Problem 4 Spectrum Plot');
    plot(w, 20*log10(H_mag), 'LineWidth', 2, 'Color', [0.1 0.5 0.7]);
    
    grid on;
    title('Magnitude Spectrum of 8th-Order All-Pole System (Vocal Tract Model)');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    xlim([0, Fs/2]); % Plot only up to Nyquist frequency (8000 Hz)
    set(gca, 'FontSize', 10);
    hold on; % Keep the plot open so the main script can add formant markers
end
%% --- BW calculation ---
function [Formants_Hz, Bandwidths_Hz] = estimate_and_plot_formants(Poles, Fs, H_mag)
% ESTIMATE_AND_PLOT_FORMANTS Calculates the formant frequencies and bandwidths 
% from the system poles and annotates the current magnitude spectrum plot.
%
%   Inputs:
%       Poles: Array of the system's poles.
%       Fs: Sampling frequency in Hz.
%       H_mag: Magnitude of the frequency response H(e^j\omega) (used for plotting limits).
%
%   Outputs:
%       Formants_Hz: Array of calculated formant frequencies.
%       Bandwidths_Hz: Array of calculated formant bandwidths.
    
    % --- Core Calculation Logic ---

    % 1. Isolate upper-half poles and sort by angle (frequency)
    Poles_upper_half = Poles(imag(Poles) > 0); 
    [~, sortIdx] = sort(angle(Poles_upper_half));
    Poles_sorted = Poles_upper_half(sortIdx);

    num_formants = length(Poles_sorted);
    Formants_Hz = zeros(num_formants, 1);
    Bandwidths_Hz = zeros(num_formants, 1);

    % --- Output Header ---
    fprintf('------------------------------------------------------------------------------------------------\n');
    fprintf('Problem 4b: Formant and Bandwidth Estimation (Fs = %d Hz)\n', Fs);
    fprintf('------------------------------------------------------------------------------------------------\n');
    fprintf(' i |     Pole (Real, Imag)      | Pole Radius (r) | Pole Angle (theta rad) | Formant F_i (Hz) | Bandwidth B_i (Hz)\n');
    fprintf('---|----------------------------|-----------------|------------------------|------------------|--------------------\n');

    % --- Loop through poles to calculate and plot ---
    H_db = 20*log10(H_mag);
    min_db = min(H_db); 
    max_db = max(H_db);

    for k = 1:num_formants
        p = Poles_sorted(k);
        r = abs(p);
        theta = angle(p);
        
        % Formant Frequency (F_k) in Hz: F_k = (theta / 2*pi) * Fs
        Fk = (theta / (2*pi)) * Fs;
        
        % Bandwidth (B_k) in Hz: B_k = - (Fs / pi) * log(r)
        Bk = - (Fs / pi) * log(r);
        
        Formants_Hz(k) = Fk;
        Bandwidths_Hz(k) = Bk;

        % Print detailed results to console
        fprintf('%2d | %8.6f + %8.6f j | %11.6f | %15.6f | %15.2f | %17.2f\n', k, real(p), imag(p), r, theta, Fk, Bk);

        % Plotting the Formant frequencies as vertical lines on the spectrum
        % NOTE: The plot_pole_spectrum function must be called beforehand to setup the figure with 'hold on'
        plot([Fk, Fk], [min_db, max_db], '--r', 'HandleVisibility','off');
    end

    % --- Annotate Plot and Final Summary ---
    
    % Labeling the Formants on the plot
    for k = 1:num_formants
        text(Formants_Hz(k), max_db - (k)*1, ['F', num2str(k), ': ', num2str(Formants_Hz(k), '%.0f'), ' Hz'], ...
            'Color', 'red', 'FontSize', 8, 'HorizontalAlignment', 'center');
    end
    legend('System Spectrum', 'Location', 'NorthEast');
    hold off;

    fprintf('------------------------------------------------------------------------------------------------\n');
    fprintf('Summary of Estimated Formant Frequencies (F1-F%d):\n[%s] Hz\n', num_formants, num2str(Formants_Hz.', '%.2f'));
    fprintf('Summary of Estimated Bandwidths (B1-B%d):\n[%s] Hz\n', num_formants, num2str(Bandwidths_Hz.', '%.2f'));

end
%% --- Plot BW ---
function fig_handle =plot_BW_poles(Poles, Fs)
% PLOT_BW_POLES Calculates formant frequencies and bandwidths from poles 
% and displays the results in a MATLAB uitable figure.
%
%   Inputs:
%       Poles: Array of the system's poles.
%       Fs: Sampling frequency in Hz.
%
%   Outputs:
%       (Creates a new figure containing a uitable)

    % --- 1. Calculate Formant and Bandwidth Data ---

    % Isolate upper-half poles and sort by angle (frequency)
    Poles_upper_half = Poles(imag(Poles) > 0); 
    [~, sortIdx] = sort(angle(Poles_upper_half));
    Poles_sorted = Poles_upper_half(sortIdx);

    num_formants = length(Poles_sorted);
    % Initialize cell array: Index, Pole, Radius, Angle, Formant, Bandwidth
    data_cell = cell(num_formants, 6);

    for k = 1:num_formants
        p = Poles_sorted(k);
        r = abs(p);
        theta = angle(p);
        
        % Formant Frequency (F_k) in Hz: F_k = (theta / 2*pi) * Fs
        Fk = (theta / (2*pi)) * Fs;
        
        % Bandwidth (B_k) in Hz: B_k = - (Fs / pi) * log(r)
        Bk = - (Fs / pi) * log(r);
        
        % Populate the cell array with formatted strings for the table
        data_cell{k, 1} = num2str(k);
        data_cell{k, 2} = sprintf('%8.6f + %8.6f j', real(p), imag(p));
        data_cell{k, 3} = sprintf('%11.6f', r);
        data_cell{k, 4} = sprintf('%15.6f', theta);
        data_cell{k, 5} = sprintf('%15.2f', Fk);
        data_cell{k, 6} = sprintf('%17.2f', Bk);
    end

    % --- 2. Table Setup and Uitable Creation ---
    
    headers = {'i (Index)', 'Pole (Real, Imag)', 'Pole Radius (r)', ...
               'Pole Angle (theta rad)', 'Formant F_i (Hz)', 'Bandwidth B_i (Hz)'};

    % Create Figure
    fig_handle = figure('Name', 'Formant and Bandwidth Results Table', 'NumberTitle', 'off', 'Color', 'w');
    
    % Create Uitable
    u = uitable(fig_handle);
    u.Data = data_cell;
    u.ColumnName = headers;
    u.RowName = {}; % Hide row numbers
    u.Units = 'normalized';
    u.Position = [0.02 0.02 0.96 0.93]; % Full figure size with margin
    u.FontSize = 10;
    
    % Set column widths for better fit
    col_widths = {50, 150, 100, 100, 100, 100}; 
    u.ColumnWidth = col_widths;
    
    % Add title to the figure
    title_str = sprintf('Formant and Bandwidth Estimation Results (Fs=%d Hz)', Fs);
    uicontrol('Style', 'text', 'String', title_str, ...
              'Units', 'normalized', 'Position', [0.1 0.95 0.8 0.05], ...
              'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'w');

    % Maximize the figure for better view if it has many columns
    set(fig_handle, 'WindowState', 'maximized');
end
%% --- Recover Poles ---
function Recovered_Poles = recover_poles_from_coeffs(A_coeffs)
% RECOVER_POLES_FROM_COEFFS Calculates the roots (poles) of the denominator 
% polynomial A(z) given its coefficients.
%
%   Input:
%       A_coeffs: Array of the denominator coefficients 
%                 A(z) = A(1) + A(2)*z^-1 + ... + A(N)*z^-(N-1), where A(1)=1.
%
%   Output:
%       Recovered_Poles: Array of the system's poles.
%
% The MATLAB 'roots' function takes the polynomial coefficients and returns 
% the roots (poles).

    Recovered_Poles = roots(A_coeffs);

    % Optional: Sort the poles for cleaner display, often by magnitude or angle
    [~, sortIdx] = sort(angle(Recovered_Poles));
    Recovered_Poles = Recovered_Poles(sortIdx);

    fprintf('-----------------------------------------------------------\n');
    fprintf('Problem 5: Recovered Poles from A(z) Coefficients\n');
    fprintf('-----------------------------------------------------------\n');
    fprintf('Coefficient Order N = %d (System Order = %d)\n', length(A_coeffs), length(A_coeffs)-1);
    fprintf('Recovered Poles (Z-Plane Roots):\n');
    for i = 1:length(Recovered_Poles)
        p = Recovered_Poles(i);
        fprintf('  p%d: %8.6f + %8.6f j\n', i, real(p), imag(p));
    end
    fprintf('-----------------------------------------------------------\n');
end
%% ---save figure to picture---
function figure_to_png(figHandle, filename, relative_save_path)
% FIGURE_TO_PNG Saves a specified MATLAB figure handle as a high-quality PNG file.
%
%   NOTE: This final version uses 'exportapp' (the recommended method for
%         figures containing UI components like uitable) to eliminate the 
%         "UI components will not be included" warning.
%
%   Inputs:
%       figHandle: Handle to the figure object.
%       filename: The base filename (e.g., 'problem1_time_domain').
%       relative_save_path: The desired save path relative to the *current* %                           function's directory (e.g., '../Data/Results/plot').
%
%   Output:
%       Saves the figure as a PNG file in the specified directory structure.
    
    % 1. Determine the current directory of this function
    current_file_path = mfilename('fullpath');
    current_dir = fileparts(current_file_path);
    
    % 2. Construct the absolute output directory path
    outputDir = fullfile(current_dir, relative_save_path);
    
    % 3. Ensure the output directory exists
    if ~exist(outputDir, 'dir')
        [success, message, ~] = mkdir(outputDir);
        if ~success
            fprintf(2, 'Error creating directory %s: %s\n', outputDir, message);
            return; 
        end
    end
    
    % 4. Construct the full file path for saving
    fullPath = fullfile(outputDir, [filename, '.png']);
    
    try
        % Force a redraw before saving to ensure everything is rendered
        drawnow; 
        
        % --- Final Robust Saving Method: Use exportapp ---
        % Using exportapp(figHandle, fullPath) is the most robust way to save 
        % figures containing uitables. It treats the figure as an application 
        % window, ensuring all UI elements are rendered correctly, and is 
        % compatible with figures containing only graphics as well.
        if exist('exportapp', 'file')
            % Use exportapp for universal, warning-free UI component export
            % The minimal syntax avoids the "Too many input arguments" error 
            % seen in the previous attempt.
            exportapp(figHandle, fullPath); 
            fprintf('Successfully saved figure (using exportapp) to: %s\n', fullPath);
        else 
            % Fallback to 'exportgraphics' if 'exportapp' is unavailable
            if exist('exportgraphics', 'file')
                 exportgraphics(figHandle, fullPath, 'Resolution', 300);
                 fprintf('Successfully saved figure (using exportgraphics fallback) to: %s\n', fullPath);
            else
                 % Final fallback for old MATLAB versions
                 set(figHandle, 'PaperPositionMode', 'auto');
                 print(figHandle, fullPath, '-dpng', '-r300');
                 fprintf('Successfully saved figure (using print fallback) to: %s\n', fullPath);
            end
        end
        
    catch ME
        % Print the actual error message that occurred during saving
        fprintf(2, 'Error saving figure %s: %s\n', filename, ME.message);
        disp('Saving failed. Check figure handle, file permissions, and MATLAB version compatibility.');
    end
end
