      subroutine voigtv(n,a,v,h)
      integer n
      real*8 a(n),v(n), h(n)
c
c this vectorizable voigt function is based on the paper by
c hui, armstrong and wray, jqsrt 19, 509 (1977). it has been
c checked agains the old landolt & boernstein standard voigt. errors
c become significant (at the < 1 % level) around the "knee" between
c the dopler core and the damping wings for a smaller than 1.e-3.
c note that, as written here, the routine has no provision for
c the trivial case a=0. the normalization is such that the integral
c is sqrt(pi); i.e., a plain exp(-x**2) should be used at a=0.
c the new landolt & boernstein gives a useful expression for the
c small but finite a case.
c
c coded by: a. nordlund/16-dec-82.
c
      complex z
      data a0/122.607931777104326/
      data a1/214.382388694706425/
      data a2/181.928533092181549/
      data a3/93.155580458138441/
      data a4/30.180142196210589/
      data a5/5.912626209773153/
      data a6/0.564189583562615/
      data b0/122.607931773875350/
      data b1/352.730625110963558/
      data b2/457.334478783897737/
      data b3/348.703917719495792/
      data b4/170.354001821091472/
      data b5/53.992906912940207/
      data b6/10.479857114260399/
	save a0,a1,a2,a3,a4,a5,a6,b0,b1,b2,b3,b4,b5,b6
c

Cf2py intent(in) a
Cf2py intent(in) v
Cf2py intent(out) h

      do 100 i=1,n
      z=cmplx(a(i),-abs(v(i)))
      h(i)=real(
     & ((((((a6*z+a5)*z+a4)*z+a3)*z+a2)*z+a1)*z+a0)
     & /(((((((z+b6)*z+b5)*z+b4)*z+b3)*z+b2)*z+b1)*z+b0)
     & )
100   continue
      return
      end
