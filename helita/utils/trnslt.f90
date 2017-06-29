! Fortran subroutines from SCATE
!
!-------------------------------------------------------------------------------
subroutine trnslt_old(mx,my,mz,nt,dx,dy,zt,ff,dxdz,dydz)

  !  Translate a scalar field to an inclined coordinate system.
  !
  !  Operation count:  10m+8a = 18 flops/pnt
  !
  !  Timing:
  !    Alliant: 48 calls = 1.79 s ->  18*48*31*31*31/1790000 = 14 Mflops
  !
  !  Update history:
  !
  !  28-oct-87/aake:  Added 'nt' parameter; reduces work in shallow case
  !  02-nov-87/aake:  Added 'zt' parameter, to allow separate zrad()
  !  06-nov-87/aake:  Split derivative loops, to loop over simplest index
  !  27-aug-89/bob:   Inverted some l,m loops to make l the inner loop
  !  31-aug-89/aake:  Collapsed loops 100 and 200 to lm loops

  implicit none
  integer,intent(in) :: mx,my,mz,nt
  real,intent(in) :: dx,dy,dxdz,dydz
  real,dimension(nt),intent(in) ::  zt
  real,dimension(mx,my,mz),intent(inout) :: ff

  real,dimension(mx,my) :: f,d
  integer :: k,l,m,n,lp,lp1,mk,mk1
  real :: xk,yk,p,q,af,bf,ad,bd

  do n=1,nt
     xk=dxdz*zt(n)/(mx*dx)
     if(abs(xk).lt.0.001.or.mx.eq.1)cycle
     xk=amod(xk,1.)
     if(xk.lt.0.) xk=xk+1.
     xk=mx*xk
     k=xk
     p=xk-k
     q=1.-p
     af=q+p*q*(q-p)
     bf=p-p*q*(q-p)
     ad=p*q*q
     bd=-p*q*p

     !  Copy input to temporary
     do m=1,my
        do l=1,mx
           f(l,m)=ff(l,m,n)
        end do
     end do

     !  Calculate derivatives by centered differences [1m+1a]
     do m=1,my
        do l=2,mx-1
           d(l,m)=0.5*(f(l+1,m)-f(l-1,m))
        end do
     end do
     do m=1,my
        d(1,m)=0.5*(f(2,m)-f(mx,m))
        d(mx,m)=0.5*(f(1,m)-f(mx-1,m))
     end do
     !
     !  Interpolate using cubic splines [4m+3a]
     !
     do l=1,mx
        lp=mod(l+k-1,mx)+1
        lp1=mod(l+1+k-1,mx)+1
        do m=1,my
           ff(l,m,n)=af*f(lp,m)+bf*f(lp1,m)+ad*d(lp,m)+bd*d(lp1,m)
           !             0.29 sec
        end do
     end do
  end do

  do n=1,nt
     yk=dydz*zt(n)/(my*dy)
     if(abs(yk).lt.0.001.or.my.eq.1)cycle
     yk=amod(yk,1.)
     if(yk.lt.0.) yk=yk+1.
     yk=my*yk
     k=yk
     p=yk-k
     q=1.-p
     af=q+p*q*(q-p)
     bf=p-p*q*(q-p)
     ad=p*q*q
     bd=-p*q*p

     !  Copy input to temporary
     do m=1,my
        do l=1,mx
           f(l,m)=ff(l,m,n)
        end do
     end do

     !  Calculate derivatives by centered differences
     do m=2,my-1
        do l=1,mx
           d(l,m)=0.5*(f(l,m+1)-f(l,m-1))
           !             0.16 sec
        end do
     end do
     do l=1,mx
        d(l,1)=0.5*(f(l,2)-f(l,my))
        d(l,my)=0.5*(f(l,1)-f(l,my-1))
     end do

     !  Interpolate using cubic splines
     do m=1,my
        mk=mod(m+k-1,my)+1
        mk1=mod(m+1+k-1,my)+1
        do l=1,mx
           ff(l,m,n)=af*f(l,mk)+bf*f(l,mk1)+ad*d(l,mk)+bd*d(l,mk1)
           !             0.18 sec -> 48*31*31*31*7/0.18 =
        end do
     end do
  end do

end subroutine trnslt_old


!-------------------------------------------------------------------------------
subroutine trnslt(mx,my,mz,nzt,dx,dy,zt,f,dxdz,dydz)

  !  Translate a scalar field to an inclined coordinate system.
  !
  !  Adapted from original routine by Nordlund and Stein

  implicit none

  integer,                   intent(in)    :: mx, my, mz, nzt
  real,                      intent(in)    :: dx, dy, dxdz, dydz
  real, dimension(nzt),      intent(in)    :: zt
  real, dimension(mx,my,mz), intent(inout) :: f
  real, dimension(mx,my)                   :: ftmp

  integer :: k, l, m, n, lm1, lp0, lp1, lp2, mm1, mp0, mp1, mp2
  real    :: xk, yk, p, q, af, bf, ad, bd, ac, bc

  real, parameter :: eps=1.0e-6


  if (abs(dxdz).gt.eps) then

     do n=1,nzt

        xk = dxdz*zt(n)/(mx*dx)
        xk = amod(xk,1.)
        if (xk.lt.0.) xk = xk + 1.
        xk = mx*xk
        k  = xk
        p  = xk-k
        k  = k + mx
        q  = 1.-p
        af = q+p*q*(q-p)
        bf = p-p*q*(q-p)
        ad = p*q*q*0.5
        bd = -p*q*p*0.5
        ac = af-bd
        bc = bf+ad

        do m=1,my
           do l=1,mx
              ftmp(l,m) = f(l,m,n)
           end do
        end do

        do l=1,mx
           lm1 = mod(l+k-2,mx)+1
           lp0 = mod(l+k-1,mx)+1
           lp1 = mod(l+k  ,mx)+1
           lp2 = mod(l+k+1,mx)+1

           do m=1,my
              f(l,m,n) = ac * ftmp(lp0,m) + bc * ftmp(lp1,m) - ad * ftmp(lm1,m) + bd * ftmp(lp2,m)
           end do
        end do

     end do

  end if


  if (abs(dydz).gt.eps) then

     do n=1,nzt

        yk = dydz*zt(n)/(my*dy)
        yk = amod(yk,1.)
        if (yk.lt.0.) yk = yk + 1.
        yk = my*yk
        k  = yk
        p  = yk - k
        k  = k + my
        q  = 1. - p
        af = q+ p*q*(q-p)
        bf = p-p*q*(q-p)
        ad = p*q*q*0.5
        bd = -p*q*p*0.5
        ac = af-bd
        bc = bf+ad

        do m=1,my
           do l=1,mx
              ftmp(l,m) = f(l,m,n)
           end do
        end do

        do m=1,my
           mm1 = mod(m+k-2,my)+1
           mp0 = mod(m+k-1,my)+1
           mp1 = mod(m+k  ,my)+1
           mp2 = mod(m+k+1,my)+1
           do l=1,mx
              f(l,m,n) = ac * ftmp(l,mp0) + bc * ftmp(l,mp1) - ad * ftmp(l,mm1) + bd * ftmp(l,mp2)
           end do
        end do

     end do

  end if

end subroutine trnslt
