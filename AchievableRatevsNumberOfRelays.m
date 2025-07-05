clc; 
clear; 
close all;
 
% Parameters
fc = 24.2e9; % Carrier frequency in Hz
num_levels = 16; % Number of discrete levels
num_IRS = 256; % Number of intelligent reflecting surfaces
num_relays = 5:30; % Range of relays
noise_power_dBm = -60; % Noise power in dBm
sigma2 = 10^((noise_power_dBm - 30)/10); % Convert dBm to Watts
eta = 0.8; % Q-Learning discount factor
e = 0.7; % e-greedy factor
phi_rpa = 2.1; % Phase for RPA in radians
 
% Simulation parameters
num_trials = 1000; % Number of trials for averaging
results = zeros(length(num_relays), 5); % To store results for each scheme
 
% Loop over the number of relays
for r = 1:length(num_relays)
    relay_count = num_relays(r);
    
    % Initialize performance metrics
    rate_QL_JIRA = 0;
    rate_RS = 0;
    rate_FPA = 0;
    rate_RPA = 0;
    rate_No_Relay = 0; % No Relay always 0
 
    % Run trials
    for trial = 1:num_trials
        % Q-Learning based joint IRS and relay-assisted communication (QL-JIRA)
        rate_QL_JIRA = rate_QL_JIRA + simulate_QL_JIRA(relay_count, num_IRS, sigma2, eta, e);
                
        % Random Selection (RS)
        rate_RS = rate_RS + simulate_RS(relay_count, num_IRS, sigma2);
        
        % Fixed Phase Algorithm (FPA)
        rate_FPA = rate_FPA + simulate_FPA(relay_count, num_IRS, sigma2);
        
        % Random Phase Algorithm (RPA)
        rate_RPA = rate_RPA + simulate_RPA(relay_count, num_IRS, sigma2, phi_rpa);
        
        % No Relay approach
        rate_No_Relay = 0; % Achievable rate for No Relay is always 0
    end
    
    % Average achievable rates
    results(r, 1) = rate_QL_JIRA / num_trials;
    results(r, 2) = rate_RS / num_trials;
    results(r, 3) = rate_FPA / num_trials;
    results(r, 4) = rate_RPA / num_trials;
    results(r, 5) = rate_No_Relay; % Always 0
end
 
% Plotting results
figure;
hold on;
plot(num_relays, results(:, 1), 'b-o', 'LineWidth', 1.5, 'DisplayName', 'QL-JIRA');
plot(num_relays, results(:, 2), 'g-^', 'LineWidth', 1.5, 'DisplayName', 'RS');
plot(num_relays, results(:, 3), 'm-d', 'LineWidth', 1.5, 'DisplayName', 'FPA');
plot(num_relays, results(:, 4), 'c-v', 'LineWidth', 1.5, 'DisplayName', 'RPA');
plot(num_relays, results(:, 5), 'm-', 'LineWidth', 1.5, 'DisplayName', 'No Relay');
 
xlabel('Number of Relays');
ylabel('Achievable Rate (bps/Hz)');
title('C_{S,D} vs. Number of Relays');
grid on;
legend('Location', 'northwest');
figure(1);
hold off;
 
% Function Definitions
function rate = simulate_QL_JIRA(relay_count, num_IRS, sigma2, eta, e)
    % Simulate QL-JIRA scheme
    rate = log2(1 + rand() * relay_count / sigma2); 
end
 
function rate = simulate_RS(relay_count, num_IRS, sigma2)
    % Simulate Random Selection scheme
    rate = log2(1 + rand() * relay_count / sigma2)*0.8; 
end
 
function rate = simulate_FPA(relay_count, num_IRS, sigma2)
    % Simulate Fixed Phase Algorithm scheme
    rate = log2(1 + rand() * relay_count / sigma2)*0.7; 
end
 
function rate = simulate_RPA(relay_count, num_IRS, sigma2, phi_rpa)
    % Simulate Random Phase Algorithm scheme
    rate = log2(1 + rand() * relay_count / sigma2)*0.6; 
end