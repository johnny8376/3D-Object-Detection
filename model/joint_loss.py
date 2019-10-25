def part_loc():
    """
    Determin internal location
    """
    [h,w,l,3]

class interal_transform():

    R = [[cos-theta, -sin-theta],[sin-thrta, cos-theta]]
    t = (p-x) * R
    O = (t-O)/[h,w,l]

    def forward(self,x):
    ntheta = -()
    p = 
    C = 
    tx = (p[0]-C[-0])*torch.cos(ntheta)-(p[1]-C[1])*torch.sin(ntheta)
    ty = (p[0]-C[-0])*torch.sin(ntheta)+(p[1]-C[1])*torch.cos(ntheta)
    Ox, Oy, Oz = tx/w+0.5, ty/l+0.5, (p-Oz)/h+0.5

    return Ox, Oy, Oz

