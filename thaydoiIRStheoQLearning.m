% IRS Network Simulation with Q-Learning
clc; clear; close all;

% Parameters
fc = 24.2e9; % Carrier frequency (Hz)
noise_power_dBm = -60; % Noise power (dBm)
noise_power = 10^(noise_power_dBm / 10); % Noise power in linear scale
eta = 0.8; % Q-Learning discount factor
e = 0.7; % E-greedy factor
num_IRS_values = [64, 128, 256, 512]; % Different IRS sizes
phi_rpa = 2.1; % Reflection phase adjustment (radians)
num_discrete_levels = 16; % Discrete phase levels
transmit_power_dBm = 0:5:30; % Transmit power (dBm)
transmit_power = 10.^(transmit_power_dBm / 10); % Transmit power (linear scale)
B = 1e6; % Bandwidth (Hz)

% Q-Learning Initialization
num_states = num_discrete_levels; % Number of states (phases)
Q_table = zeros(num_states, length(transmit_power)); % Q-table
achievable_rates = zeros(length(transmit_power), length(num_IRS_values)); % Achievable rates

% Loop over IRS configurations
for irs_idx = 1:length(num_IRS_values)
    num_IRS = num_IRS_values(irs_idx); % Number of IRS elements
    
    % Loop over transmit power levels
    for tp_idx = 1:length(transmit_power)
        P_tx = transmit_power(tp_idx); % Transmit power for this iteration
        
        % Simulate IRS channel and achievable rate
        h_IRS = sqrt(num_IRS) * (randn(1, num_states) + 1j * randn(1, num_states)) / sqrt(2); % IRS channel
        SNR_IRS = (P_tx * abs(h_IRS).^2) / noise_power; % Signal-to-noise ratio
        rate_IRS = B * log2(1 + SNR_IRS); % Achievable rate
        
        % Q-Learning to optimize phase
        [max_rate, optimal_phase_idx] = max(rate_IRS); % Select best phase
        achievable_rates(tp_idx, irs_idx) = max_rate; % Store the max achievable rate
        
        % Update Q-table using Q-Learning
        Q_table(optimal_phase_idx, tp_idx) = ...
            (1 - eta) * Q_table(optimal_phase_idx, tp_idx) + eta * max_rate;
    end
end

% Plot results
figure;
hold on; grid on;
for irs_idx = 1:length(num_IRS_values)
    plot(transmit_power_dBm, achievable_rates(:, irs_idx), '-o', ...
        'LineWidth', 1.5, 'DisplayName', sprintf('IRS (%d elements)', num_IRS_values(irs_idx)));
end

% Configure plot
xlabel('Transmit Power (dBm)');
ylabel('Achievable Rate (bps/Hz)');
title('Achievable Rate vs Transmit Power for IRS with Q-Learning');
legend('Location', 'northwest');
set(gca, 'FontSize', 12);
hold off;