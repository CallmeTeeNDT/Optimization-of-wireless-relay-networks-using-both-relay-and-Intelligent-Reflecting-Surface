clc;
clear;
close all;
 
% System parameters
fc = 24.2e9; % Carrier frequency (Hz)
num_levels = 16; % Number of discrete levels for IRS
num_IRS = 256; % Number of intelligent reflecting surfaces
num_relays = 5:5:30; % Number of relays
noise_power_dBm = -60; % Noise power (dBm)
noise_power = 10^(noise_power_dBm / 10) * 1e-3; % Convert to Watts
epsilon = 0.7; % e-greedy factor
eta = 0.8; % Q-Learning discount factor
phi_rpa = 2.1; % Fixed phase for RPA (radians)
 
% Transmit power (dBm) range
P_dBm = 0:5:70;
P = 10.^(P_dBm / 10) * 1e-3; % Convert to Watts
 
% Initialize achievable rates
rate_QL_JIRA = zeros(length(num_relays), length(P));
rate_RS = zeros(length(num_relays), length(P));
rate_FPA = zeros(length(num_relays), length(P));
rate_RPA = zeros(length(num_relays), length(P));
rate_NoRelay = zeros(1, length(P));
 
for idx_relay = 1:length(num_relays)
    R = num_relays(idx_relay); % Number of relays
 
    % Channel gains (random initialization for simplicity)
    h_S_IRS = abs(randn(1, num_IRS)); % Source to IRS
    h_S_R = abs(randn(R, 1)); % Source to Relays
    h_IRS_D = abs(randn(1, num_IRS)); % IRS to Destination
    h_IRS_R = abs(randn(R, num_IRS)); % IRS to Relays
    h_R_IRS = abs(randn(R, num_IRS)); % Relays to IRS
    h_R_D = abs(randn(R, 1)); % Relays to Destination
 
    % Q-Learning initialization
    Q = zeros(R, num_levels); % Q-table
    num_iterations = 100000; % Increased iterations for better convergence
 
    for p_idx = 1:length(P)
        Pt = P(p_idx); % Transmit power (W)
 
        % QL-JIRA
        for iter = 1:num_iterations
            % Explore: Choose random relay and phase shift
            relay_state = randi(R);
            if rand < epsilon
                action = randi(num_levels); % Random phase shift
            else
                % Exploit: Choose best action from Q-table
                [~, action] = max(Q(relay_state, :));
            end
 
            % Compute reward (Achievable Rate)
            phase_shift = 2 * pi * (action - 1) / num_levels;
            effective_channel1 = abs(h_S_R(relay_state) + (h_IRS_R(relay_state, :) * (exp(1j * phase_shift) .* h_S_IRS.')).');
            effective_channel2 = abs(h_R_D(relay_state) + (h_R_IRS(relay_state, :) * (exp(1j * phase_shift) .* h_IRS_D.')).');
 
            % Combined effective channel for two time slots
            effective_channel = sqrt(effective_channel1.^2 + effective_channel2.^2);
            rate = 0.5 * min(log2(1 + (Pt * sum(effective_channel.^2) / noise_power))); 
 
            % Update Q-table with reward
            Q(relay_state, action) = Q(relay_state, action) + ...
                eta * (rate - Q(relay_state, action));
        end
 
        % Optimal relay and phase shift selection for QL-JIRA
        [~, optimal_action] = max(max(Q, [], 2));
        optimal_phase = 2 * pi * (optimal_action - 1) / num_levels;
        effective_channel1 = abs(h_S_R(optimal_action) + (h_IRS_R(optimal_action, :) * (exp(1j * optimal_phase) .* h_S_IRS.')).');
        effective_channel2 = abs(h_R_D(optimal_action) + (h_R_IRS(optimal_action, :) * (exp(1j * optimal_phase) .* h_IRS_D.')).');
        effective_channel = sqrt(effective_channel1.^2 + effective_channel2.^2);
        rate_QL_JIRA(idx_relay, p_idx) = 0.5 * min(log2(1 + (Pt * sum(effective_channel.^2) / noise_power))); 
 
        % RS (Random Relay Selection)
        random_relay = randi(R);
        random_phase = rand * 2 * pi;
        effective_channel1 = abs(h_S_R(random_relay) + (h_IRS_R(random_relay, :) * (exp(1j * random_phase) .* h_S_IRS.')).');
        effective_channel2 = abs(h_R_D(random_relay) + (h_R_IRS(random_relay, :) * (exp(1j * random_phase) .* h_IRS_D.')).');
        effective_channel = sqrt(effective_channel1.^2 + effective_channel2.^2);
        rate_RS(idx_relay, p_idx) = 0.5 * min(log2(1 + (Pt * sum(effective_channel.^2) / noise_power))); 
 
        % FPA (Fixed Phase Alignment)
        random_relay = randi(R);
        fixed_phase = 0; % Assume optimal phase alignment
        effective_channel1 = abs(h_S_R(random_relay) + (h_IRS_R(random_relay, :) * (exp(1j * fixed_phase) .* h_S_IRS.')).');
        effective_channel2 = abs(h_R_D(random_relay) + (h_R_IRS(random_relay, :) * (exp(1j * fixed_phase) .* h_IRS_D.')).');
        effective_channel = sqrt(effective_channel1.^2 + effective_channel2.^2);
        rate_FPA(idx_relay, p_idx) = 0.5 * min(log2(1 + (Pt * sum(effective_channel.^2) / noise_power))); 
 
        % RPA (Random Phase Alignment)
        effective_channel1 = abs(h_S_R(random_relay) + (h_IRS_R(random_relay, :) * (exp(1j * phi_rpa) .* h_S_IRS.')).');
        effective_channel2 = abs(h_R_D(random_relay) + (h_R_IRS(random_relay, :) * (exp(1j * phi_rpa) .* h_IRS_D.')).');
        effective_channel = sqrt(effective_channel1.^2 + effective_channel2.^2);
        rate_RPA(idx_relay, p_idx) = 0.5 * min(log2(1 + (Pt * sum(effective_channel.^2) / noise_power)));
    end
end
 
% No Relay case
for p_idx = 1:length(P)
    Pt = P(p_idx);
    effective_channel = abs(h_S_IRS .* exp(1j * phi_rpa) .* h_IRS_D);
    rate_NoRelay(p_idx) = 0.5 * min(log2(1 + (Pt * sum(effective_channel.^2) / noise_power)));
end
 
% Plot results
figure(2);
hold on;
plot(P_dBm, mean(rate_QL_JIRA, 1), 'o-', 'LineWidth', 1.5, 'DisplayName', 'QL-JIRA');
plot(P_dBm, mean(rate_RS, 1), 's-', 'LineWidth', 1.5, 'DisplayName', 'RS');
plot(P_dBm, mean(rate_FPA, 1), '^-', 'LineWidth', 1.5, 'DisplayName', 'FPA');
plot(P_dBm, mean(rate_RPA, 1), 'x-', 'LineWidth', 1.5, 'DisplayName', 'RPA');
plot(P_dBm, rate_NoRelay, 'm-', 'LineWidth', 1.5, 'DisplayName', 'No Relay');
 
xlabel('Transmit Power (dBm)');
ylabel('Achievable Rate (bps/Hz)');
grid on;
legend show;
title('Achievable Rate vs Transmit Power');
