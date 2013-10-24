module tridiag

use iso_c_binding, only: c_double, c_int

implicit none

contains

subroutine ftridiag(a, b, c, x, n) bind(c)
    
    integer(c_int), intent(in) :: n
    real(c_double),dimension(n),intent(in) :: b
    real(c_double),dimension(n),intent(inout) :: x
    real(c_double),dimension(n-1),intent(in) :: a
    real(c_double),dimension(n-1),intent(inout) :: c
    
    real(c_double) :: m
    integer i
        
    c(1) = c(1) / b(1)
    x(1) = x(1) / b(1)
    do i = 1,n-2
        m = 1.0D0 / (b(i+1) - a(i)*c(i))
        c(i+1) = c(i+1) * m
        x(i+1) = x(i+1) - a(i)*x(i)
        x(i+1) = x(i+1) * m
    enddo
    x(n) = (x(n) - a(n-1)*x(n-1)) / (b(n) - a(n-1)*c(n-1))
    do i = n-1,1,-1
        x(i) = x(i) - c(i) * x(i+1)
    enddo
end subroutine ftridiag

end module
