clc;
clear;
close all;

% Parameters
fc = 24.2e9; % Carrier frequency (Hz)
c = 3e8; % Speed of light (m/s)
lambda = c / fc; % Wavelength (m)
levels = 16; % Number of discrete levels
num_IRS = 256; % Number of intelligent reflecting surfaces
noise_power = 10^(-60/10); % Noise power in linear scale (W)
eta = 0.8; % Q-Learning discount factor
epsilon = 0.7; % e-greedy factor
phi_rpa = 2.1; % Radians

relay_range = 5:5:30; % Number of relays to test
transmit_power_dBm = 0:5:30; % Transmit power in dBm
transmit_power = 10.^(transmit_power_dBm / 10) / 1000; % Convert to Watts

% Placeholder for results
achievable_rate = zeros(length(relay_range), length(transmit_power));

% Q-Learning Parameters
Q_table = zeros(levels, num_IRS); % Initialize Q-table

for r_idx = 1:length(relay_range)
    num_relays = relay_range(r_idx);

    for p_idx = 1:length(transmit_power)
        P_tx = transmit_power(p_idx); % Current transmit power

        % Calculate path loss (simplified free-space model)
        d = 100; % Distance between transmitter and receiver (m)
        path_loss = (lambda / (4 * pi * d))^2;

        % Simulate Q-Learning for relay selection
        for episode = 1:1000
            state = randi(levels); % Random initial state

            for step = 1:10
                if rand < epsilon
                    action = randi(num_IRS); % Explore
                else
                    [~, action] = max(Q_table(state, :)); % Exploit
                end

                % Reward computation
                relay_gain = 10^(num_relays / 10); % Approximate relay gain
                SNR = (P_tx * path_loss * relay_gain) / noise_power;
                rate = log2(1 + SNR); % Achievable rate (bps/Hz)
                reward = rate; % Reward proportional to rate

                % Update Q-table
                next_state = randi(levels); % Random next state
                Q_table(state, action) = Q_table(state, action) + ...
                    eta * (reward + max(Q_table(next_state, :)) - Q_table(state, action));

                state = next_state; % Move to next state
            end
        end

        % Compute average achievable rate
        SNR_final = (P_tx * path_loss * 10^(num_relays / 10)) / noise_power;
        achievable_rate(r_idx, p_idx) = log2(1 + SNR_final);
    end
end

% Plot results
figure;
for r_idx = 1:length(relay_range)
    plot(transmit_power_dBm, achievable_rate(r_idx, :), '-o', 'LineWidth', 2.5);
    hold on;
end

grid on;
title('Achievable Rate vs Transmit Power with Relay Network');
xlabel('Transmit Power (dBm)');
ylabel('Achievable Rate (bps/Hz)');
legend(arrayfun(@(x) sprintf('%d Relays', x), relay_range, 'UniformOutput', false), 'Location', 'Best');
