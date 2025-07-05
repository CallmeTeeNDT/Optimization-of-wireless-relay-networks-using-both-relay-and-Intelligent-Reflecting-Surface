clc; close all; clear;

%% ------------------------------------------------------------------------
%   PHẦN 1: CÁC THIẾT LẬP CHUNG CHO MÔ HÌNH KÊNH VÀ IRS
%   Tham số hệ thống
num_samples = 100000;                   % Số mẫu ngẫu nhiên
num_relays  = 5;                        % Số lượng relay
fc = 24.2e9;                            % Tần số sóng mang (Hz)
h_UT = 1;                               % Chiều cao User Terminal (m)
P_dBm = 0:5:70;                         % Công suất truyền (dBm)
P = 10.^(P_dBm / 10) * 1e-3;            % Đổi dBm -> Watts
xi = 1e-3;                              % Ngưỡng hội tụ
sigma2_dBm = -60;                       % Công suất nhiễu (dBm)
sigma2 = 10^(sigma2_dBm / 10) * 1e-3;   % Đổi dBm -> Watts
N_IRS = 256;                            % Số phần tử IRS
num_levels = 16;                        % Số mức lượng tử pha IRS (ví dụ)
d_example = 10;                         % Khoảng cách ví dụ (m)
phi_rpa = 2.1;                          % Phase for RPA (radians)

% Hàm tính Path Loss (PL)
calc_PL_LOS  = @(d, fc) 32.4 + 21*log10(d) + 20*log10(fc/1e9);
calc_PL_NLOS = @(d, fc, h_UT) 22.4 + 35.5*log10(d) + 21.3*log10(fc/1e9) - 0.3*(h_UT - 1.5);
calc_PL      = @(d, fc, h_UT) max(calc_PL_LOS(d, fc), calc_PL_NLOS(d, fc, h_UT));

% Tính K-factor dựa trên Path Loss tại khoảng cách d_example
PL_LOS_value  = calc_PL_LOS(d_example, fc);
PL_NLOS_value = calc_PL_NLOS(d_example, fc, h_UT);
K_dB    = PL_NLOS_value - PL_LOS_value; 
K_factor = 10^(K_dB / 10);

% Hàm tạo kênh Rician ngẫu nhiên
rician_channel = @(K, M, N) sqrt(K/(K+1)) ...
    + sqrt(1/(K+1)).*(randn(M,N) + 1i*randn(M,N))/sqrt(2);

%% ------------------------------------------------------------------------
%   PHẦN 2: MÔ PHỎNG TẠO CÁC KÊNH NGẪU NHIÊN
% - h_s_Ri_phase   : kênh từ Nguồn (S) -> Relay i -> IRS (nếu cần)
% - h_s_IRS_phase  : kênh từ Nguồn (S) -> IRS
% - h_IRS_Ri_phase : kênh từ IRS -> Relay i
% - h_IRS_D_phase  : kênh từ IRS -> Đích (D)
% - h_Ri_D_phase   : kênh từ Relay i -> Đích (D)
% - h_Ri_IRS_phase : kênh từ Relay i -> IRS
h_s_Ri_phase   = rician_channel(K_factor, num_samples, N_IRS);
h_s_IRS_phase  = rician_channel(K_factor, num_samples, N_IRS);
h_IRS_Ri_phase = rician_channel(K_factor, num_samples, N_IRS);
h_IRS_D_phase  = rician_channel(K_factor, num_samples, N_IRS);
h_Ri_D_phase   = rician_channel(K_factor, num_samples, N_IRS);
h_Ri_IRS_phase = rician_channel(K_factor, num_samples, N_IRS);

%% ------------------------------------------------------------------------
%   PHẦN 3: ALGORITHM 2 (Reward Matrix Procedure)
h_RS_Ri = abs( randn(num_relays,1) + 1i*randn(num_relays,1) );
RW = rewardMatrixProcedure(h_RS_Ri);

%% ------------------------------------------------------------------------
%   PHẦN 4: ALGORITHM 3 (Q-Learning Based Relay Selection)
%   Các tham số Q-learning (bạn điều chỉnh theo nhu cầu)
numEpisodes = 10000;  % Số vòng lặp học
epsilon     = 0.7;    % Hệ số epsilon-greedy
alpha       = 0.1;    % Tốc độ học
gamma       = 0.8;    % Hệ số chiết khấu

% Khởi tạo Q-table
Q = zeros(num_relays, num_relays);

for episode = 1:numEpisodes
    % Chọn state ngẫu nhiên
    s_t = randi([1 num_relays]);  
    % Chọn action ngẫu nhiên
    a_t = randi([1 num_relays]);
    
    % Epsilon-greedy
    if rand > epsilon
        [~, a_t] = max(Q(s_t,:)); 
    end
    
    % Quan sát reward từ ma trận RW
    R_t = RW(s_t, a_t);
    
    % Giả sử state kế tiếp s_{t+1} = a_t (hoặc có thể random tuỳ bạn)
    s_tplus1 = a_t;
    
    % Cập nhật Q(s_t, a_t)
    Q(s_t, a_t) = Q(s_t, a_t) + alpha * ( R_t + gamma * max(Q(s_tplus1,:)) );
end

% Tính relay tốt nhất dựa trên giá trị trung bình
[~, bestRelayAll] = max( mean(Q,2) );
fprintf('Relay tối ưu (theo trung bình Q) là: Relay %d\n', bestRelayAll);

%% ------------------------------------------------------------------------
%   PHẦN 5: TỐI ƯU PHA IRS 
% Khởi tạo pha ngẫu nhiên ban đầu cho IRS 
Phi_1 = exp(1i * rand(N_IRS, 1) * 2 * pi);
Phi_2 = exp(1i * rand(N_IRS, 1) * 2 * pi);
m = 0;  % Đếm số vòng lặp

for idx_P = 1:length(P)
    P_i = P(idx_P);  
    % Tính tốc độ truyền ban đầu (Time slot 1)
    C_1 = log2(1 + ( P_i .* abs( sum( h_s_Ri_phase ...
                        + (h_IRS_Ri_phase .* Phi_1.') .* h_s_IRS_phase, 2 ) ).^2 )/sigma2);
    % Tính tốc độ truyền ban đầu (Time slot 2)
    C_2 = log2(1 + ( P_i .* abs( sum( h_Ri_D_phase ...
                        + (h_IRS_D_phase .* Phi_2.') .* h_Ri_IRS_phase, 2 ) ).^2 )/sigma2);

    % ------ Vòng lặp tối ưu pha IRS cho khe thời gian 1 ------
    while true
        C_1_old = C_1;
        for n = 1:N_IRS
            w = sum( h_IRS_Ri_phase(:,n) .* h_s_IRS_phase(:,n), 1 );
            phi_opt_1 = -angle(w);
            Phi_1(n) = exp(1i * phi_opt_1);
        end
        C_1 = log2(1 + ( P_i .* abs( sum( h_s_Ri_phase ...
                        + (h_IRS_Ri_phase .* Phi_1.') .* h_s_IRS_phase, 2 ) ).^2 )/sigma2);
        
        if abs(mean(C_1 - C_1_old)) <= xi
            break;
        end
        m = m + 1;
    end
    % ------ Vòng lặp tối ưu pha IRS cho khe thời gian 2 ------
    while true
        C_2_old = C_2;
        for n = 1:N_IRS
            w = sum( h_IRS_D_phase(:,n) .* h_Ri_IRS_phase(:,n), 1 );
            phi_opt_2 = -angle(w);
            Phi_2(n) = exp(1i * phi_opt_2);
        end
        C_2 = log2(1 + ( P_i .* abs( sum( h_Ri_D_phase ...
                        + (h_IRS_D_phase .* Phi_2.') .* h_Ri_IRS_phase, 2 ) ).^2 )/sigma2);
        
        if abs(mean(C_2 - C_2_old)) <= xi
            break;
        end
        m = m + 1;
    end
end

% Kết quả cuối cùng
Phi_opt_1 = Phi_1;   
C_1_star  = mean(C_1); 
Phi_opt_2 = Phi_2;   
C_2_star  = mean(C_2);
C_s_d_star = (1/2)*min(C_1_star, C_2_star);
C_s_d = mean(C_s_d_star);

disp('Optimal phase angles of 1:');
disp(Phi_opt_1);
disp('Optimal phase angles of 2:');
disp(Phi_opt_2);
disp('Optimal achievable rate:');
disp(C_s_d);

function RW = rewardMatrixProcedure(h_RS_Ri_vec)
    n = length(h_RS_Ri_vec);
    RW = zeros(n,n);
    for i = 1:n
        for j = 1:n
            RW(i,j) = h_RS_Ri_vec(j) / h_RS_Ri_vec(i);
        end
    end
end