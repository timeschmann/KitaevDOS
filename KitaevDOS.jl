# -------------------------------------------------------------------
# JULIA TOOL TO CALCULATE MAJORANA DENSITIES OF STATES
# FOR DIFFERENT 2D AND 3D KITAEV MODELS
# (written for Julia 0.6)
# Author: Tim Eschmann
# Date: 2019-11-11
# Modified for new version of LatticePhysics: 2019-01-31
# -------------------------------------------------------------------

using PyCall

# Get unit cell size
function get_uc_length(lat)
    if lat == "6a"
        N = 2
    elseif lat == "8a"
        N = 6
    elseif lat == "8b"
        N = 6
    elseif lat == "8n"
        N = 16
    elseif lat == "10a"
        N = 4
    elseif lat == "10b"
        N = 4
    elseif lat == "10c"
        N = 12
    elseif lat == "10d"
        N = 8
    end
    return N
end

# Generate Bloch matrix for lattice 'lat' with couplings 'J' at k-point 'k'
function get_ham(k,J,lat)
    if lat == "6a"
        f(k,J) = 2*(J[1]*exp(2*im*k[1]*pi) + J[2]*exp(2*im*k[2]*pi) + J[3])
        ham = [0 im*f(k,J); -im*conj(f(k,J)) 0]
    elseif lat == "8a"
        A1(k,J) = exp(2*im*k[1]*pi)*J[2]
        A2(k,J) = exp(-2*im*k[2]*pi)*J[3]
        A3(k,J) = exp(-2*im*k[3]*pi)*J[1]
        B = [im*J[2] im*A2(k,J) 0; im*J[3] 0 im*A1(k,J); im*J[1] im*J[2] im*J[3]]
        C1 = [0 0 0; 0 0 im*A3(k,J); 0 -im*conj(A3(k,J)) 0]
        C2 = [0 -im*A3(k,J) 0; im*conj(A3(k,J)) 0 0; 0 0 0]
        ham = [C2 B; transpose(conj(B)) C1]
    elseif lat == "8b"
        a13(k,J) = exp(-2*im*(k[1]+k[3])*pi)*J[2]
        a2(k,J) = exp(2*im*k[2]*pi)*J[1]
        a3(k,J) = exp(-2*im*k[3]*pi)*J[1]
        B2 = [im*J[3] 0 -im*a13(k,J); -im*J[2] im*J[3] 0; -im*a2(k,J) -im*J[2] im*J[3]]
        C1 = [0 -im*a3(k,J) 0; im*conj(a3(k,J)) 0 0 ; 0 0 0]
        C2 = [0 0 0; 0 0 im*a3(k,J); 0 -im*conj(a3(k,J)) 0]
        ham = [C1 B2; transpose(conj(B2)) C2]
    elseif lat == "8n"
        b1(k,J) = exp(-2*im*k[1]*pi)
        b13(k,J) = exp(-2*im*(k[1]-k[3])*pi)
        b2(k,J) = exp(2*im*k[2]*pi)
        b23(k,J) = exp(2*im*(k[2]-k[3])*pi)
        b3(k,J) = exp(2*im*k[3]*pi)
        D = [im*J[3] im*J[1] 0 0 0 0 -im*J[2] 0; 0 -im*J[2] im*J[3] 0 0 0 0 -im*J[1]; 0 0 -im*J[1] im*J[2] 0 0 -im*b3(k,J)*J[3] 0; 0 0 0 -im*J[3] im*J[1] 0 0 -im*b2(k,J)*J[2]; 0 0 0 0 -im*J[2] im*J[3] im*b2(k,J)*J[1] 0; im*J[2] 0 0 0 0 -im*J[1] 0 im*b13(k,J)*J[3]; 0 im*b23(k,J)*J[3] 0 -im*b1(k,J)*J[1] 0 im*J[2] 0 0; -im*J[1] 0 -im*b1(k,J)*J[2] 0 -im*conj(b3(k,J))*J[3] 0 0 0]    
        E = zeros(8,8)
        ham = [E D; transpose(conj(D)) E]
    elseif lat == "10a"
        c1(k,J) = exp(-2*im*k[1]*pi)*J[3]
        c2(k,J) = exp(-2*im*k[2]*pi)*J[1]
        c3(k,J) = exp(-2*im*k[3]*pi)*J[1]
        ham = [0 -im*c2(k,J) -im*J[2] -im*c1(k,J); im*conj(c2(k,J)) 0 -im*J[3] im*J[2]; im*J[2] im*J[3] 0 -im*c3(k,J); im*conj(c1(k,J)) -im*J[2] im*conj(c3(k,J)) 0]
    elseif lat == "10b"
        A13(k,J) = exp(-2*im*k[3]*pi)*(J[1] + exp(2*im*k[1]*pi) * J[2])
        A4(k,J) = J[1] + exp(2*im*k[2]*pi) * J[2]
        B = [0 0; 0 0]
        C = [im*J[3] im*A13(k,J); im*A4(k,J) im*J[3]]
        ham = [B C; transpose(conj(C)) B]
    elseif lat == "10c"
        d1(k,J) = J[1] + exp(-2*im*k[1]*pi)*J[2]
        d2(k,J) = exp(-2*im*k[3]*pi)*J[3]
        d3(k,J) = exp(-2*im*k[2]*pi)*J[2]
        d4(k,J) = exp(2*im*k[1]*pi)*J[2]
        d5(k,J) = -J[1] + exp(-2*im*k[1]*pi)*J[2]
        d6(k,J) = exp(2*im*(k[1]+k[2])*pi)*J[2]
        A = [im*d1(k,J) 0 -im*d2(k,J) 0 0 0; im*J[3] -im*J[1] 0 0 im*d3(k,J) 0; 0 im*J[3] -im*J[1] 0 0 im*d4(k,J); 0 0 0 im*d5(k,J) 0 -im*d2(k,J); 0 im*J[2] 0 im*J[3] im*J[1] 0; 0 0 im*d6(k,J) 0 im*J[3] -im*J[1]]
        ham = [zeros(6,6) A; transpose(conj(A)) zeros(6,6)]
    elseif lat == "10d"
        A_ = im*[0 J[3]*exp(2*im*k[2]*pi) J[1] J[2]*exp(-2*im*k[3]*pi); J[3]*exp(-2*im*k[1]*pi) 0 J[2] J[1]; J[1] J[2] 0 J[3]; J[2]*exp(-2*im*k[3]*pi) J[1] J[3]*exp(2*im*(k[1]-k[2])*pi) 0]
        ham = [zeros(4,4) A_; transpose(conj(A_)) zeros(4,4)]
    end 
    return 2*ham
end

function main()
    # Define lattice, couplings and number of k-points
    lats = ["8a"]
    J = [1/3., 1/3., 1/3.]
    ni = 400
    nj = 400
    nk = 400

    a1 = [1,0,0]
    a2 = [0,1,0]
    a3 = [0,0,1]
    
    # 2D:
    #bzone = [i*a1 + j*a2 for i=0:1/ni:1 for j=0:1/nj:1]

    # 3D:
    bzone = [i*a1 + j*a2 + k*a3 for i=0:1/ni:1 for j=0:1/nj:1 for k=0:1/nk:1]
    #println(bzone)
    
    for lat in lats
        N = get_uc_length(lat)
        #evals = Array{Float64,2}((ni+1)*(nj+1),N) #2D
        evals = Array{Float64,2}((ni+1)*(nj+1)*(nk+1),N) #3D
        ct = 1
        for k in bzone
            evals[ct,:] = eigvals(get_ham(k,J,lat))
            ct += 1
            if ct % 10000 == 0
                println("$(lat):  $(ct) / $(size(bzone,1))\n")
            end
        end
    
        #println(minimum(evals), " ", maximum(evals))

        open("dos$(lat).txt", "w") do f
            for i in 1:size(evals,1)
                for j in 1:size(evals,2)
                    write(f, "$(evals[i,j])\n")
                end
            end 
        end
    end

end

main()
