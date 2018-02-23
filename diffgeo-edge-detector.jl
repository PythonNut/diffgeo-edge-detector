using ImageFiltering, TestImages
using ImageView
using Base.Cartesian
img = testimage("mandrill");

scales = exp.(linspace(0,log(50),40))
img_scaled = cat(3,(imfilter(img, reflect(Kernel.gaussian(t))) for t in scales)...)
L = float.(Colors.Gray.(img_scaled))

Ay = Array(parent(Kernel.ando5()[1]))
Ax = Array(parent(Kernel.ando5()[2]))

function image_derivative_kernel(ker, kers...)
    for kernel in kers
        ker = conv2(ker, kernel)
    end
    return ker
end

function image_derivative(I, kers...)
    kernel = image_derivative_kernel(kers...)
    return imfilter(I, reflect(centered(kernel)))
end

function image_scale_derivative(L, kers...)
    return mapslices(Ls->image_derivative(Ls, kers...), L, (1,2))
end

Lx = image_scale_derivative(L, Ax)
Ly = image_scale_derivative(L, Ay)

Lxx = image_scale_derivative(L, Ax, Ax)
Lxy = image_scale_derivative(L, Ax, Ay)
Lyy = image_scale_derivative(L, Ay, Ay)

Lxxx = image_scale_derivative(L, Ax, Ax, Ax)
Lxxy = image_scale_derivative(L, Ax, Ax, Ay)
Lxyy = image_scale_derivative(L, Ax, Ay, Ay)
Lyyy = image_scale_derivative(L, Ay, Ay, Ay)

# Lx, Ly = imgradients(L, KernelFactors.ando3)
for (i, scale) in enumerate(scales)
    Lx[:,:,i] *= scale ^ 0.5
    Ly[:,:,i] *= scale ^ 0.5
end

Lgrad = sqrt.(Lx.^2 + Ly.^2)

# Lxy, Lxx = imgradients(Lx, KernelFactors.ando3)
# Lyy, Lyx = imgradients(Ly, KernelFactors.ando3)
# Lxxy, Lxxx = imgradients(Lxx, KernelFactors.ando3)
# Lxyy, Lxyx = imgradients(Lxy, KernelFactors.ando3)
# Lyyy, Lyyx = imgradients(Lyy, KernelFactors.ando3)

Lvv = Lx.^2.*Lxx + 2Lx.*Ly.*Lxy + Ly.^2.*Lyy
Lvvv = Lx.^3.*Lxxx + 3Lx.^2.*Ly.*Lxxy + 3Lx.*Ly.^2.*Lxyy + Ly.^3.*Lyyy

dLdt = zeros(Bool, size(L))
for i in 2:(length(scales)-1)
    dLdt[:,:,i] = (L[:,:,i-1] .< L[:,:,i]) .& (L[:,:,i] .> L[:,:,i+1])
end

Lvv_pos = Lvv .> 0
Lvv_zeros = zeros(Bool, size(Lvv))
for (i, scale) in enumerate(scales)
    for x in 2:(size(Lvv)[1]-1)
        for y in 2:(size(Lvv)[2]-1)
            neighbors = Lvv_pos[x-1:x+1, y-1:y+1, i]
            if (Lvv_pos[x,y,i] && !all(neighbors)) ||
                (!Lvv_pos[x,y,i] && any(neighbors))
                Lvv_zeros[x,y,i] = true
            end
        end
    end
end

Ledges = (Lvvv .< 0) .& Lvv_zeros

zr, slicedata = roi(Lgrad, (1,2))
gd = imshow_gui((512, 512), slicedata, (1,2))
imshow(gd["frame"][1,1], gd["canvas"][1,1], (Lgrad./maximum(Lgrad)), nothing, zr, slicedata)
imshow(gd["frame"][1,2], gd["canvas"][1,2], Ledges, nothing, zr, slicedata)
showall(gd["window"])
