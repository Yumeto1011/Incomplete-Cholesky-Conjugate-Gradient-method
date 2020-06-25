module iccg

using LinearAlgebra
    export ICCG

    # 不完全コレスキー分解(incomplete Cholesky decomposition)
    #  - 対称行列A(n×n)を下三角行列(L:Lower triangular matrix)と対角行列の積(LDL^T)に分解する
    #  - l_ii = 1
    #  - L: i > jの要素が非ゼロで対角成分は1
    #  - 行列Aの値が0である要素に対応する部分を0とする
    # in : A n×nの対称行列
    # out :  L 対角成分が1の下三角行列
    # out :  d 対角行列の対角成分要素のベクトル
    function IC(A)
        n = size(A)[1]
        L = zeros(Float64, size(A))
        d = zeros(Float64, 1, n)
        
        L[1,1] = 1.0
        d[1] = A[1,1]
        
        for i in 2:n
            # 下三角行列 L を求める　（対角成分は0で、対応する行列Aの要素が0であるときはLの要素を0とする）
            
            if A[i,1] == 0.0
                L[i, 1] = 0.0
            else
                L[i, 1] = A[i, 1]/d[1]
            end
            
            for j in 2:i-1
                if A[i,j] == 0.0
                    L[i,j] = 0.0
                else
                    ldl = 0.0
                    for k in 1:(j-1)
                        if A[i, k] == 0.0
                            ldl += 0.0
                        elseif A[j, k] == 0.0
                            ldl += 0.0
                        else
                            ldl += L[i, k]*d[k]*L[j, k]
                        end
                    end
                    L[i,j] = (A[i,j]-ldl)/d[j]
                end
            end
            
            L[i, i] = 1.0
            ###
            
            # 対角行列 D の対角成分要素のベクトル d を求める
            l2d = 0.0
            for k in 1:(i-1)
                l2d += L[i,k]^2*d[k]
            end
            
            d[i] = A[i,i] - l2d
            ###
        end
        return L,d
    end

    # 不完全コレスキー分解分解した行列A(n×n)から前進代入と後退代入によりAx=bを解く
    # A = LDL'とすると　Ax=b -> LDL'x = b
    #                  まず、y=L'x として LDy=b をyについて前進代入で解く（LDが下三角行列だから）
    #                  次に、L'x=y をxについて後退代入で解く（L'が上三角行列だから）
    # in : Aを不完全コレスキー分解分解した行列　L,Dの対角要素のベクトルｄ
    # in : b 右辺ベクトル
    function LDLtSolver(L,d,b)
        n = size(L)[1]
        
        # 前進代入でy=L'x  として LDy=b をyについて解く
        y = zeros(Float64, size(b))
        
        y[1] = b[1]/d[1]
        
        for i in 2:n
            dly = 0.0
            for j in 1:(i-1)
                dly += d[j]*L[i,j]*y[j]
            end
            y[i] = (b[i] - dly)/d[i]
        end
        ###
        
        # 後退代入で L'x=y をxについて解く
        x = zeros(Float64, size(b))
        
        x[n] = y[n]
        
        for i in (n-1):-1:1
            lx = 0.0
            for j in (i+1):n
                lx += L[j,i]*x[j]
            end
            x[i] = y[i] - lx
        end
        ###
        
        return x
    end

    # 不完全コレスキー分解付き共役勾配法 (ICCG) によりAx=bを解く
    # in A n×n正値対称行列
    # in x 初期近似解
    # in b 右辺ベクトル
    # in max_iter 最大反復数(反復終了後,実際の反復数を返す)
    # in eps 許容誤差(反復終了後,実際の誤差を返す) 
    function ICCG(A, x, b, max_iter, eps)
        # 不完全コレスキー分解で L,D を求める
        L, d = IC(A)
        
        # 初期残差を求める
        r = b - A*x
        
        # 前進代入、後退代入（不完全コレスキー分解様に修正（LU分解のLをICのLDに、LU分解のUをLの転置に要素を見直した））で初期修正方向ベクトルを求める
        p = r
        
        # loop開始
        iter = 1
        while iter<max_iter
            # 修正方向係数を計算と近似解の更新の準備
            y = A*p
            
            # 修正方向係数の計算準備
            z = LDLtSolver(L,d,r)
            
            # 修正係数を計算
            alpha = r'*z/(p'*y)
            
            # 近似解の更新
            x = x + alpha*p
            
            # 次のステップの残差ベクトルr_k+1を計算
            r_kp1 = r - alpha*y
            
            # 残差の大きさを求める
            e = sqrt(r_kp1'*r_kp1)
                    
            # 収束判定
            if e<eps
                break
            else
                # 次のステップのz_k+1を前進代入、後退代入で計算
                z_kp1 = LDLtSolver(L,d,r_kp1)
                
                # 修正方向係数を計算
                beta = r_kp1'*z_kp1/(r'*z)
                
                # 修正方向ベクトルを修正
                p = z_kp1 + beta*p
                
                # 残差ベクトルr_k+1 -> r 、残差ベクトルのチルダrt_k+1 -> rt と置きなおす
                r = deepcopy(r_kp1)
                
                iter += 1
            end
        end
        
        return x
    end

end
