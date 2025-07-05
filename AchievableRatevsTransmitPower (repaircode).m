clc; close all; clear;
%% ------------------------------------------------------------------------
%   PHẦN 1: CÁC THIẾT LẬP CHUNG CHO MÔ HÌNH KÊNH VÀ IRS
num_samples = 10000;                  
num_relays  = 5;                       
fc = 24.2;                           
h_UT = 1;                              
P_dBm = 0:5:70;                        
P = 10.^(P_dBm / 10) * 1e-3;           
xi = 1e-3;                             
sigma2_dBm = -60;                      
sigma2 = 10^(sigma2_dBm / 10) * 1e-3;  
N_IRS = 256;                           
num_levels = 16;                       
d_example = 10;                        
phi_rpa = 2.1;                         

% Khoảng cách
d_S_IRS = 15;  
d_IRS_D = 15;  
d_S_Ri  = randi([10, 30], num_relays, 1);
d_IRS_Ri= randi([5, 20],  num_relays, 1);
d_Ri_D  = randi([10, 40], num_relays, 1);

% Hàm tính Path Loss (PL)
calc_PL_LOS  = @(d, fc) 32.4 + 21*log10(d) + 20*log10(fc);
calc_PL_NLOS = @(d, fc, h_UT) 22.4 + 35.5*log10(d) + 21.3*log10(fc) - 0.3*(h_UT - 1.5);
calc_PL      = @(d, fc, h_UT) max(calc_PL_LOS(d, fc), calc_PL_NLOS(d, fc, h_UT));

% K-factor
PL_LOS_value  = calc_PL_LOS(d_example, fc);
PL_NLOS_value = calc_PL_NLOS(d_example, fc, h_UT);
K_dB    = PL_NLOS_value - PL_LOS_value; 
K_factor = 10^(K_dB / 10);

% Hàm tạo kênh Rician
rician_channel = @(K, M, N) sqrt(K/(K+1)) ...
    + sqrt(1/(K+1)).*(randn(M,N) + 1i*randn(M,N))/sqrt(2);

%% ------------------------------------------------------------------------
%   PHẦN 2: TẠO KÊNH NGẪU NHIÊN
PL_S_IRS = calc_PL(d_S_IRS, fc, h_UT);
scale_S_IRS = 10^(-PL_S_IRS/10);
h_S_IRS_phase = sqrt(scale_S_IRS)*rician_channel(K_factor, num_samples, N_IRS);

PL_IRS_D = calc_PL(d_IRS_D, fc, h_UT);
scale_IRS_D = 10^(-PL_IRS_D/10);
h_IRS_D_phase = sqrt(scale_IRS_D)*rician_channel(K_factor, num_samples, N_IRS);

h_s_Ri_phase   = zeros(num_samples, N_IRS, num_relays);
h_IRS_Ri_phase = zeros(num_samples, N_IRS, num_relays);
h_Ri_IRS_phase = zeros(num_samples, N_IRS, num_relays);
h_Ri_D_phase   = zeros(num_samples, N_IRS, num_relays);

for i = 1:num_relays
    PL_S_Ri  = calc_PL(d_S_Ri(i), fc, h_UT);
    scale_S_Ri = 10^(-PL_S_Ri/10);
    h_s_Ri_phase(:,:,i) = sqrt(scale_S_Ri)*rician_channel(K_factor, num_samples, N_IRS);
    
    PL_IRS_Ri  = calc_PL(d_IRS_Ri(i), fc, h_UT);
    scale_IRS_Ri = 10^(-PL_IRS_Ri/10);
    h_IRS_Ri_phase(:,:,i) = sqrt(scale_IRS_Ri)*rician_channel(K_factor, num_samples, N_IRS);
    h_Ri_IRS_phase(:,:,i) = sqrt(scale_IRS_Ri)*rician_channel(K_factor, num_samples, N_IRS);
    
    PL_Ri_D  = calc_PL(d_Ri_D(i), fc, h_UT);
    scale_Ri_D = 10^(-PL_Ri_D/10);
    h_Ri_D_phase(:,:,i) = sqrt(scale_Ri_D)*rician_channel(K_factor, num_samples, N_IRS);
end

%% ------------------------------------------------------------------------
%   PHẦN 3: ALGORITHM 2 (Reward Matrix Procedure)
h_RS_Ri = abs(randn(num_relays,1) + 1i*randn(num_relays,1));
RW = rewardMatrixProcedure(h_RS_Ri);

%% ------------------------------------------------------------------------
%   PHẦN 4: ALGORITHM 3 (Q-Learning Based Relay Selection)
numEpisodes = 10000;
epsilon     = 0.7;
alpha       = 0.1;
gamma       = 0.8;

Q = zeros(num_relays, num_relays);

for episode = 1:numEpisodes
    s_t = randi([1 num_relays]);
    if rand > epsilon
        [~, a_t] = max(Q(s_t,:)); 
    else
        a_t = randi([1 num_relays]);
    end
    
    R_t = RW(s_t, a_t);
    s_tplus1 = a_t;
    
    Q(s_t, a_t) = Q(s_t, a_t) + alpha*( R_t + gamma*max(Q(s_tplus1,:)) );
end

[~, bestRelay] = max(Q(1,:));
fprintf('Relay tối ưu (Q-learning) = %d\n', bestRelay);
[~, bestRelayAll] = max(mean(Q,2));
fprintf('Relay tối ưu (trung bình Q) = %d\n', bestRelayAll);

%% ------------------------------------------------------------------------
%   PHẦN 5: MÔ PHỎNG 5 PHƯƠNG PHÁP
Cs_d_QL_Jira = zeros(size(P));
Cs_d_RS      = zeros(size(P));
Cs_d_FPA     = zeros(size(P));
Cs_d_RPA     = zeros(size(P));
Cs_d_NoRelay = zeros(size(P));

for idx_P = 1:length(P)
    P_i = P(idx_P);
    
    %% -------- 1) QL-JIRA --------
    Phi_1_QL = exp(1i * rand(N_IRS, 1) * 2*pi);
    while true
        C_1_old = computeRateTimeSlot(P_i, Phi_1_QL, ...
            h_s_Ri_phase, h_IRS_Ri_phase, h_Ri_IRS_phase, sigma2, bestRelayAll);
        % Cập nhật pha
        for n = 1:N_IRS
            w = sum( h_IRS_Ri_phase(:,n,bestRelayAll) .* ...
                     h_s_Ri_phase(:,n,bestRelayAll), 1 );
            phi_opt_1 = -angle(w);
            Phi_1_QL(n) = exp(1i*phi_opt_1);
        end
        C_1_new = computeRateTimeSlot(P_i, Phi_1_QL, ...
            h_s_Ri_phase, h_IRS_Ri_phase, h_Ri_IRS_phase, sigma2, bestRelayAll);
        if abs(mean(C_1_new - C_1_old)) <= xi
            break;
        end
    end
    
    Phi_2_QL = exp(1i * rand(N_IRS, 1) * 2*pi);
    while true
        C_2_old = computeRateTimeSlot2(P_i, Phi_2_QL, ...
            h_Ri_D_phase, h_IRS_D_phase, h_Ri_IRS_phase, sigma2, bestRelayAll);
        % Cập nhật pha
        for n = 1:N_IRS
            w = sum( h_IRS_D_phase(:,n) .* ...
                     h_Ri_IRS_phase(:,n,bestRelayAll), 1 );
            phi_opt_2 = -angle(w);
            Phi_2_QL(n) = exp(1i*phi_opt_2);
        end
        C_2_new = computeRateTimeSlot2(P_i, Phi_2_QL, ...
            h_Ri_D_phase, h_IRS_D_phase, h_Ri_IRS_phase, sigma2, bestRelayAll);
        if abs(mean(C_2_new - C_2_old)) <= xi
            break;
        end
    end
    
    C_1_QL = computeRateTimeSlot(P_i, Phi_1_QL, ...
        h_s_Ri_phase, h_IRS_Ri_phase, h_Ri_IRS_phase, sigma2, bestRelayAll);
    C_2_QL = computeRateTimeSlot2(P_i, Phi_2_QL, ...
        h_Ri_D_phase, h_IRS_D_phase, h_Ri_IRS_phase, sigma2, bestRelayAll);
    Cs_d_QL_Jira(idx_P) = 0.5 * min(mean(C_1_QL), mean(C_2_QL));  
    
        %% -------- 2) Random Selection (RS) --------
    randomRelay = randi([1 num_relays]);
    Phi_1_RS = exp(1i*rand(N_IRS,1)*2*pi);
    Phi_2_RS = exp(1i*rand(N_IRS,1)*2*pi);
    
    C_1_RS = computeRateTimeSlot(P_i, Phi_1_RS, ...
        h_s_Ri_phase, h_IRS_Ri_phase, h_Ri_IRS_phase, sigma2, randomRelay);
    C_2_RS = computeRateTimeSlot2(P_i, Phi_2_RS, ...
        h_Ri_D_phase, h_IRS_D_phase, h_Ri_IRS_phase, sigma2, randomRelay);
    Cs_d_RS(idx_P) = 0.5 * min(mean(C_1_RS), mean(C_2_RS));
    
        %% -------- 3) Fixed Phase Algorithm (FPA) --------
    Phi_1_FPA = exp(1i*rand(N_IRS,1)*2*pi);
    while true
        C_1_old = computeRateTimeSlot(P_i, Phi_1_FPA, ...
            h_s_Ri_phase, h_IRS_Ri_phase, h_Ri_IRS_phase, sigma2, bestRelay);
        for n = 1:N_IRS
            w = sum( h_IRS_Ri_phase(:,n,bestRelay) .* ...
                     h_s_Ri_phase(:,n,bestRelay), 1 );
            phi_opt_1 = -angle(w);
            Phi_1_FPA(n) = exp(1i*phi_opt_1);
        end
        C_1_new = computeRateTimeSlot(P_i, Phi_1_FPA, ...
            h_s_Ri_phase, h_IRS_Ri_phase, h_Ri_IRS_phase, sigma2, bestRelay);
        if abs(mean(C_1_new - C_1_old)) <= xi
            break;
        end
    end
    Phi_2_FPA = exp(1i * phi_rpa)*ones(N_IRS,1);
    
    C_1_FPA = computeRateTimeSlot(P_i, Phi_1_FPA, ...
        h_s_Ri_phase, h_IRS_Ri_phase, h_Ri_IRS_phase, sigma2, bestRelay);
    C_2_FPA = computeRateTimeSlot2(P_i, Phi_2_FPA, ...
        h_Ri_D_phase, h_IRS_D_phase, h_Ri_IRS_phase, sigma2, bestRelay);
    Cs_d_FPA(idx_P) = 0.5 * min(mean(C_1_FPA), mean(C_2_FPA));
    
        %% -------- 4) Random Phase Algorithm (RPA) --------
    Phi_1_RPA = exp(1i*rand(N_IRS,1)*2*pi);
    Phi_2_RPA = exp(1i*rand(N_IRS,1)*2*pi);
    C_1_RPA = computeRateTimeSlot(P_i, Phi_1_RPA, ...
        h_s_Ri_phase, h_IRS_Ri_phase, h_Ri_IRS_phase, sigma2, bestRelay);
    C_2_RPA = computeRateTimeSlot2(P_i, Phi_2_RPA, ...
        h_Ri_D_phase, h_IRS_D_phase, h_Ri_IRS_phase, sigma2, bestRelay);
    Cs_d_RPA(idx_P) = 0.5 * min(mean(C_1_RPA), mean(C_2_RPA));
    
        %% -------- 5) No Relay Approach --------
    Phi_NoRelay = exp(1i*rand(N_IRS,1)*2*pi);
    C_NoRelay = computeRateNoRelay(P_i, Phi_NoRelay, ...
        h_S_IRS_phase, h_IRS_D_phase, sigma2);
    Cs_d_NoRelay(idx_P) = mean(C_NoRelay);
end

%% VẼ ĐỒ THỊ
figure; hold on; grid on;
plot(P_dBm, Cs_d_QL_Jira, '-o','LineWidth',2, 'DisplayName','QL-JIRA');
plot(P_dBm, Cs_d_RS,      '-x','LineWidth',2, 'DisplayName','RS');
plot(P_dBm, Cs_d_FPA,     '-s','LineWidth',2, 'DisplayName','FPA');
plot(P_dBm, Cs_d_RPA,     '-d','LineWidth',2, 'DisplayName','RPA');
plot(P_dBm, Cs_d_NoRelay, '-^','LineWidth',2, 'DisplayName','No Relay');
xlabel('Transmit Power (dBm)');
ylabel('Achievable Rate (bps/Hz)');
legend show;
title('Achievable Rate vs. Transmit Power');

%% ========================= HÀM PHỤ =========================
function RW = rewardMatrixProcedure(h_RS_Ri_vec)
    n = length(h_RS_Ri_vec);
    RW = zeros(n,n);
    for ii = 1:n
        for jj = 1:n
            RW(ii,jj) = h_RS_Ri_vec(jj) / h_RS_Ri_vec(ii);
        end
    end
end

% (Time slot 1)
function C_1 = computeRateTimeSlot(P_i, Phi, ...
                h_s_Ri_phase, h_IRS_Ri_phase, ...
                h_Ri_IRS_phase, sigma2, relayIndex)
    eff_channel = h_s_Ri_phase(:,:,relayIndex) ...
                + ( h_IRS_Ri_phase(:,:,relayIndex).*Phi.' ) ...
                .* h_Ri_IRS_phase(:,:,relayIndex);
    gain = norm(eff_channel,2).^2;  
    C_1 = log2(1 + (P_i.*gain)/sigma2);
end

% (Time slot 2) 
function C_2 = computeRateTimeSlot2(P_i, Phi, ...
                h_Ri_D_phase, h_IRS_D_phase, ...
                h_Ri_IRS_phase, sigma2, relayIndex)
    eff_channel = h_Ri_D_phase(:,:,relayIndex) ...
                + ( h_IRS_D_phase.*Phi.' ) ...
                .* h_Ri_IRS_phase(:,:,relayIndex);
    gain = norm(eff_channel,2).^2;
    C_2 = log2(1 + (P_i.*gain)/sigma2);
end

% No Relay
function C_no = computeRateNoRelay(P_i, Phi, ...
                h_S_IRS_phase, h_IRS_D_phase, sigma2)
    eff_channel = h_S_IRS_phase.*(Phi.').*h_IRS_D_phase;
    gain = norm(eff_channel,2).^2;
    C_no = log2(1 + (P_i.*gain)/sigma2);
end