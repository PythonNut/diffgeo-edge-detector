using ImageFiltering
using TestImages
using ImageView
using SymPy

# Parameters
gamma = 1/2
scales = exp.(linspace(0,log(50),40))

# Load the image
img = float.(Colors.Gray.(testimage("mandrill")))

# Define derivative convolution matrices
Dy = Array(parent(Kernel.ando5()[1]))
Dx = Array(parent(Kernel.ando5()[2]))

function combine_kernels(kers...)
    return reduce(conv2, kers)
end

function convolve_image(I, kers...)
    kernel = combine_kernels(kers...)
    return imfilter(I, reflect(centered(kernel)))
end

function convolve_scale_space(L, kers...)
    return mapslices(
        scale_slice -> convolve_image(scale_slice, kers...),
        L,
        (1,2)
    )
end

function convolve_gaussian(img, sigma)
    # The dimension of the convolution matrix
    length = 8*ceil(Int, sigma) + 1
    return imfilter(img, reflect(Kernel.gaussian((sigma, sigma), (length, length))))
end

L = cat(3, (convolve_gaussian(img, sigma) for sigma in scales)...)

Lx = convolve_scale_space(L, Dx)
Ly = convolve_scale_space(L, Dy)

Lxx = convolve_scale_space(Lx, Dx)
Lxy = convolve_scale_space(Lx, Dy)
Lyy = convolve_scale_space(Ly, Dy)

Lxxx = convolve_scale_space(Lxx, Dx)
Lxxy = convolve_scale_space(Lxx, Dy)
Lxyy = convolve_scale_space(Lxy, Dy)
Lyyy = convolve_scale_space(Lyy, Dy)

Lxxxx = convolve_scale_space(Lxxx, Dx)
# Lxxxy = convolve_scale_space(Lxxx, Dy)
Lxxyy = convolve_scale_space(Lxxy, Dy)
# Lxyyy = convolve_scale_space(Lxyy, Dy)
Lyyyy = convolve_scale_space(Lyyy, Dy)

t = Sym("t")
E = t^gamma
Et, Ett = diff(E, t), diff(E, t, t)

Lvv = Lx.^2.*Lxx + 2Lx.*Ly.*Lxy + Ly.^2.*Lyy
Lvvv = Lx.^3.*Lxxx + 3Lx.^2.*Ly.*Lxxy + 3Lx.*Ly.^2.*Lxyy + Ly.^3.*Lyyy

ELt, ELtt = zeros(L), zeros(L)
for (i, scale) in enumerate(scales)
    ELt[:,:,i] = float(Et(scale)) * L[:,:,i] + (Lxx[:,:,i] + Lyy[:,:,i])/2 * float(E(scale))
    ELtt[:,:,i] = float(Ett(scale)) * L[:,:,i] + float(Et(scale))*(Lxx[:,:,i] + Lyy[:,:,i]) + (Lxxxx[:,:,i] + 2Lxxyy[:,:,i] + Lyyyy[:,:,i])/4 * float(E(scale))
end


# March ourselves some cubes
voxel_visited = zeros(Bool, (x->x-1).(size(L)))

function linear_interpolate(v1, v2)
    return abs(v2)/(abs(v1) + abs(v2))
end

function marching_cubes(x, y, t)
    corners = (x:x+1, y:y+1, z:z+1)

    # Note: Maybe they don't need to be in the same corner
    if !any((view(Lvvv, corners...) .< 0) .& (view(ELtt, corners...) .< 0))
        return
    end

    const edges = [
        # bottom edges
        ((0,0,0), (0,1,0)),
        ((0,1,0), (1,1,0)),
        ((1,1,0), (1,0,0)),
        ((1,0,0), (0,0,0)),

        # side edges
        ((0,0,0), (0,0,1)),
        ((0,1,0), (0,1,1)),
        ((1,1,0), (1,1,1)),
        ((1,0,0), (1,0,1)),

        # top edges
        ((0,0,1), (0,1,1)),
        ((0,1,1), (1,1,1)),
        ((1,1,1), (1,0,1)),
        ((1,0,1), (0,0,1)),
    ]

    const faces = [
        (())
    ]

    Z1 = view(Lvv, corners...)
    Z2 = view(ELt, corners...)
    Z1_crossings = []
    Z2_crossings = []

    # Find all sign crossings w/ linear interpolation
    for (a, b) in edges
        if sign(Z1[a...]) != sign(Z1[b...])
            push!(Z1_crossings, (a,b)=>linear_interpolate(Z1[a...], Z1[b...]))
        end

        if sign(Z2[a...]) != sign(Z2[b...])
            push!(Z2_crossings, (a,b)=>linear_interpolate(Z2[a...], Z2[b...]))
        end
    end

    # Reject if there are more than two crossings for either invariant
    if length(Z1_crossings) != 2 || length(Z2_crossings) != 2
        return
    end

    # Ensure the edges
    for dim in 1:3
        if length(unique(
            map(x->x.first[dim], Z1_crossings),
            map(x->x.second[dim], Z1_crossings),
            map(x->x.first[dim], Z2_crossings),
            map(x->x.second[dim], Z2_crossings))) == 2
            return
        end
    end


end
