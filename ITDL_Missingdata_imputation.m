clear;
stressdata = readmatrix('sensor1-2.csv');
stressdata_matrix = stressdata(:450,:)
stress_tensor = reshape(trafficdata,30,15,22)
A = stress_tensor
x = 30;
y = 15;
z = 22;
N = 2;
iteration = 1;
pho=1;
beta = 0.8;
maxitem = 10000;
lm_1 = 1e-8;
lm_2 = 1e-8;
MAE = zeros(N, 1)
RMSE = zeros(N,1);
MAPE = zeros(N,1);
rsquare = zeros(N,1);
singleMAE = zeros(N,22)
for n = 1:1:N
    TsingleMAE = zeros(1,22)
    TRMSE = 0;
    TMAPE = 0;
    Trsquare = 0;
    TMAE = 0;
    for jj = 1:1:iteration
        k=0.3+0.05*n;           % 缺失率
        k2=1-k;             % 非0率
        W_S= randsrc(x,y*z,[1 0;k2 k]);
        W=reshape(W_S,x,y,z);   % W为残缺矩阵
        W_r=ones(x,y,z);
        W_r=W_r-W;
        A_r=tensor(A.*W_r);      % A_r为残缺位置对应的真值
        A_q=tensor(A.*W);        % A_q为残缺张量

        A = tensor(A);               % 变为Tensor张量
        A1 = tenmat(A_q,1);          %张量沿mode-n展开   A1-A4为矩阵
        A2 = tenmat(A_q,2);
        A3 = tenmat(A_q,3);

        [UU1, TT1, VV1] = svdsketch(A1.data,0.5);      % 奇异值分解
        [UU2, TT2, VV2] = svdsketch(A2.data,0.7);
        [UU3, TT3, VV3] = svdsketch(A3.data);

        D1 = diag(TT1);
        r1 = 1;
        pho1_current = D1(1);
        pho1_sum = sum(D1(:));
        pho1 = pho1_current/pho1_sum;
        while pho1 < pho
            r1 = r1+1;
            pho1_current = pho1_current+D1(r1);
            pho1 = pho1_current/pho1_sum;
        end

        D2 = diag(TT2);
        r2 = 1;
        pho2_current = D2(1);
        pho2_sum = sum(D2(:));
        pho2 = pho2_current/pho2_sum;
        while pho2 < pho
            r2 = r2+1;
            pho2_current = pho2_current+D2(r2);
            pho2 = pho2_current/pho2_sum;
        end

        D3 = diag(TT3);
        r3 = 1;
        pho3_current = D3(1);
        pho3_sum = sum(D3(:));
        pho3 = pho3_current/pho3_sum;
        while pho3 < pho
            r3 = r3+1;
            pho3_current = pho3_current+D3(r3);
            pho3 = pho3_current/pho3_sum;
        end

        U1 = UU1(:,:);
        U2 = UU2(:,:);
        U3 = UU3(:,:);

        S = ttm(A_q, {U1', U2', U3'});
        B = ttm(S, {U1, U2, U3});
        L = W.*B;
        L1 = tenmat(L,1);
        L2 = tenmat(L,2);
        L3 = tenmat(L,3);

        S1 = tenmat(S,1);
        S2 = tenmat(S,2);
        S3 = tenmat(S,3);

        deta_s = (1+lm_2)*ttm(L-A_q, {U1', U2', U3'});
        temp_1 = double((1+lm_2)*(L1-A1)*(kron(U3, U2))*(S1'));
        temp_2 = double((1+lm_2)*(L2-A2)*(kron(U3, U1))*(S2'));
        temp_3 = double((1+lm_2)*(L3-A3)*(kron(U2, U1))*(S3'));

        deta_u1 = temp_1;
        deta_u2 = temp_2;
        deta_u3 = temp_3;

        for step = 1:maxitem
            S = S-lm_1*deta_s;
            S1 = tenmat(S,1);
            S2 = tenmat(S,2);
            S3 = tenmat(S,3);

            U1 = U1-lm_1*deta_u1;
            U2 = U2-lm_1*deta_u2;
            U3 = U3-lm_1*deta_u3;

            B = ttm(S, {U1, U2, U3});
            L = W.*B;
            L_r = W_r.*B;
            L1 = tenmat(L,1);
            L2 = tenmat(L,2);
            L3 = tenmat(L,3);

            deta_s = beta*deta_s+(1-beta)*((1+lm_2)*ttm(L-A_q, {U1', U2', U3'}));
            temp_1_1 = double((1+lm_2)*(L1-A1)*(kron(U3, U2))*(S1'));
            temp_2_1 = double((1+lm_2)*(L2-A2)*(kron(U3, U1))*(S2'));
            temp_3_1 = double((1+lm_2)*(L3-A3)*(kron(U2, U1))*(S3'));

            deta_u1 = beta*deta_u1+(1-beta)*temp_1_1;
            deta_u2 = beta*deta_u2+(1-beta)*temp_2_1;
            deta_u3 = beta*deta_u3+(1-beta)*temp_3_1;
            Es = double(tenmat(A_q,1));
            LF = double(tenmat(L,1));
            EF = abs(Es-LF);
            loss1 = EF/(x*y*z*k2);
            loss = norm(loss1,'fro');
            if loss < 0.005
                break;
            end
        end
        B = tensor(B);
        B_r=B.*W_r;
        A_2r = double(tenmat(A_r,1));
        B_2r = double(tenmat(B_r,1));
        tot_m = A_2r;
        avg = sum(A_2r(:))/(x*y*z*k);
        temp_mape=0;
        for i=1:x
            for j=1:y*z
                if A_2r(i,j)~=0
                    tot_m(i,j)=tot_m(i,j)-avg;
                    temp_mape=abs((A_2r(i,j)-B_2r(i,j))/A_2r(i,j))+temp_mape;
                end
            end
        end
        reg_m = (A_2r-B_2r).*(A_2r-B_2r);
        tot_m = tot_m.*tot_m;
        SSreg=sum(reg_m(:));
        SStot=sum(tot_m(:));
        Trsquare = 1-(SSreg/SStot) + Trsquare;
        TRMSE = sqrt(sum(sum((A_2r-B_2r).^2))/(x*y*z*k)) + TRMSE;
        TMAPE = temp_mape/(x*y*z*k)  + TMAPE;
        TMAE = sum(sum(abs(A_2r-B_2r)))/(x*y*z*k) +TMAE
        for ii=1:1:22
            E = abs(A_2r(:,(ii-1)*y+1:ii*y)-B_2r(:,(ii-1)*y+1:ii*y));
            A_1r = A_2r(:,(ii-1)*y+1:ii*y);
            num = 0;
            for jjj=1:1:x
                for kk=1:1:y
                    if A_1r(jjj,kk) ~= 0
                        num = num + 1;
                    end
                end
            end
            TsingleMAE(1,ii) = sum(E(:))/(num) + TsingleMAE(1,ii);
        end
    end
        MAE(n,1) = TMAE/iteration;
        RMSE(n,1) = TRMSE/iteration;
        MAPE(n,1) = TMAPE/iteration;
        rsquare(n,1) = Trsquare / iteration;
        singleMAE(n,:) = TsingleMAE / iteration;
end