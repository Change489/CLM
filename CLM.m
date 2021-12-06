function [theta, b, Z, V, fvalue1,fvalue2] = CLM(Y, A, lambda, beta, gamma)
%%% Demo code for CLM method. Details please refer to our paper.

%%%%%%%%%  You need to initialize parameters %%%%%%%%%
Maxiter = 80;
rho = 1e-5;
max_mu = 1e6;
mu = 1.1;
tol = 1e-8;
tol_inter = 1e-6;
rate_theta = 0.1;
rate_b = 0.001;
maxiter = Maxiter;
iter = 0;
maxiter_inter = 100;
T = length(A);
m = size(A{1,1}(:,:,1),1);

theta = cell(T,1);
Orig_theta = cell(T,1);
b=zeros(T,1);
Z = cell(T,1);
Z1 = cell(T,1);
Z2 = cell(T,1);
V = cell(T,1);
V1 = cell(T,1);
V2 = cell(T,1);C = cell(T,1);
V3 = cell(T,1);

U1 = cell(T,1);U1_Check = cell(T,1);
U2 = cell(T,1);U2_Check = cell(T,1);
U3 = cell(T,1);U3_Check = cell(T,1);
U4 = cell(T,1);U4_Check = cell(T,1);
U5 = cell(T,1);U5_Check = cell(T,1);
U6 = cell(T,1);U6_Check = cell(T,1);
U7 = cell(T,1);U7_Check = cell(T,1);
U8 = cell(T,1);U8_Check = cell(T,1);


for i=1:T
    theta{i,1} = zeros(m,m);
    Orig_theta{i,1} = zeros(m,m);
    Z{i,1} = zeros(m,m); 
    Z1{i,1} = zeros(m,m);
    Z2{i,1} = zeros(m,m);
    V{i,1} = zeros(m,m);
    V1{i,1} = zeros(m,m);
    V2{i,1} = zeros(m,m);C{i,1} = zeros(m,m);
    V3{i,1} = zeros(m,m);
    U1{i,1} = zeros(m,m);U1_Check{i,1} = zeros(m,m);
    U2{i,1} = zeros(m,m);U2_Check{i,1} = zeros(m,m);
    U3{i,1} = zeros(m,m);U3_Check{i,1} = zeros(m,m);
    U4{i,1} = zeros(m,m);U4_Check{i,1} = zeros(m,m);
    U5{i,1} = zeros(m,m);U5_Check{i,1} = zeros(m,m);
    U6{i,1} = zeros(m,m);U6_Check{i,1} = zeros(m,m);
    U7{i,1} = zeros(m,m);U7_Check{i,1} = zeros(m,m);
    U8{i,1} = zeros(m,m);U8_Check{i,1} = zeros(m,m);
end
jj=1;

while iter<maxiter
    iter = iter +1;

    for i=1:T
        iter_inter = 0;
        while iter_inter<maxiter_inter
            iter_inter = iter_inter+1;
            theta_Pre = theta{i,1};
            
            Obj_Pre = Compute_Logistic_Obj(theta_Pre,A{i,1},Y{i,1},b(i,1));
            Obj_Pre = Obj_Pre + trace(U8{i,1}'*(theta_Pre-V{i,1}-Z{i,1})) + rho*norm(theta_Pre-V{i,1}-Z{i,1},'fro')^2/2;
            
            grad_theta = logistic_grad_W(theta{i,1},A{i,1},Y{i,1},b(i,1));
            grad_theta = grad_theta + U8{i,1}+ rho*(theta{i,1}-V{i,1}-Z{i,1});
            theta{i,1} = theta{i,1} - rate_theta*grad_theta;
            theta{i,1} = theta{i,1}-diag(diag(theta{i,1}));
            Obj =  Compute_Logistic_Obj(theta{i,1},A{i,1},Y{i,1},b(i,1))+ ...
                trace(U8{i,1}'*(theta{i,1}-V{i,1}-Z{i,1})) + rho*norm(theta{i,1}-V{i,1}-Z{i,1},'fro')^2/2;
            if norm(theta{i,1}-theta_Pre,'fro')/norm(theta_Pre,'fro') <= tol_inter
                break;
            end

            
        end
    end

    for i=1:T
        V{i,1} = (V1{i,1}+V2{i,1}+V3{i,1}'+V3{i,1}-Z{i,1}+theta{i,1})/5+...
            (U8{i,1}-U4{i,1}-U5{i,1}-U6{i,1}'-U7{i,1})/(5*rho);
        V{i,1} = V{i,1} - diag(diag(V{i,1}));
    end

    for i=1:T
        Z{i,1} = (U8{i,1}-U1{i,1}-U2{i,1}'-U3{i,1})/(4*rho)+...
            (Z1{i,1}+Z2{i,1}'+Z2{i,1}+theta{i,1}-V{i,1})/4;
        Z{i,1} = Z{i,1} - diag(diag(Z{i,1}));
    end

    for i=1:T
        V1{i,1} = solve_l1(V{i,1}+U4{i,1}/rho,beta/rho);
    end
    
    Co_hub = [];Co_hub_l21 = [];
    for i=1:T
        C{i,1} = U5{i,1}/rho+V{i,1};
        Co_hub = [Co_hub;C{i,1}];
    end
    Co_hub_l21 = solve_l1l2(Co_hub, gamma/rho);
    for i=1:T
        C{i,1} = Co_hub_l21((1+(i-1)*m):(m*i),:);
    end

    for i=1:T
        V3{i,1} = (U6{i,1}+U7{i,1})/(2*rho)+(V{i,1}'+V{i,1})/2;
    end
    

    for i=1:T
        Z1{i,1} = solve_l1(Z{i,1}+U1{i,1}/rho,lambda/rho);
    end

    for i=1:T
        Z2{i,1} = (U2{i,1}+U3{i,1})/(2*rho)+(Z{i,1}'+Z{i,1})/2;
    end


    for i=1:T
        iter_inter = 0;
        while iter_inter<maxiter_inter
            iter_inter = iter_inter+1;
            BPre   = b(i,1);
            Obj_Preb = Compute_Logistic_Obj(theta{i,1},A{i,1},Y{i,1},BPre);
            grad_b = logistic_grad_b(theta{i,1},A{i,1},Y{i,1},b(i,1));
            b(i,1) = b(i,1) - rate_b*grad_b;
            Obj_b = Compute_Logistic_Obj(theta{i,1},A{i,1},Y{i,1},b(i,1));
            if sum(abs(Obj_Preb-Obj_b))<=tol_inter
                break;
            end
        end
        
    end
    
%%% you need to updata auxillary variable and check Convergence

end


%%%additional functions

function grad = logistic_grad_W(W,X,Y,delta)
Num = size(Y,1);
grad = zeros(size(X(:,:,1)));
for i=1:Num
    grad = grad + exp(-Y(i,1)*(trace(W'*X(:,:,i))+delta))*(-Y(i,1)*X(:,:,i))/...
        (1+exp(-Y(i,1)*(trace(W'*X(:,:,i))+delta)));
end
grad = grad/Num;


function b = logistic_grad_b(W,X,Y,delta)
Num = size(Y,1);
b = 0;
for i=1:Num
    b = b + exp(-Y(i,1)*(trace(W'*X(:,:,i))+delta))*(-Y(i,1))/...
        (1+exp(-Y(i,1)*(trace(W'*X(:,:,i))+delta)));
end
b = b/Num;



function objective_value = Compute_Logistic_Obj(W,X,Y,b)
objective_value = 0;
Num = size(Y,1);
for i=1:Num
    objective_value = objective_value + log( 1+exp(-Y(i,1)*(trace(W'*X(:,:,i))+b)) );
end
objective_value = objective_value/Num;

function [E] = solve_l1(W,lambda)

if(lambda<0)
    disp('There input are error')
end
E=zeros(size(W));
temp=abs(W)-lambda;
temp(temp<=0)=0;
E=sign(W).*temp;


function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end


function [x] = solve_l2(w,lambda)
nw = norm(w,2);
if nw>lambda
    x = (1-lambda/nw)*w;
else
    x = zeros(length(w),1);
end