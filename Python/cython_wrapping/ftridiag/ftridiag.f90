module tridiag

use iso_c_binding, only: c_double, c_int

implicit none

contains

! The expression bind(c) tells the compiler to
! make the naming convention in the object file
! match the naming convention here.
! This will be a subroutine since it does not
! return any values.
subroutine ftridiag(a, b, c, x, n) bind(c)
    
!   Here we declare the types for the inputs.
!   Thisis where we use the c_double and c_int types.
!   The 'dimension' statement tells the compiler that
!   the argument is an array of the given shape.
    integer(c_int), intent(in) :: n
    real(c_double),dimension(n),intent(in) :: b
    real(c_double),dimension(n),intent(inout) :: x
    real(c_double),dimension(n-1),intent(in) :: a
    real(c_double),dimension(n-1),intent(inout) :: c
    
!   Two temporary varaibles.
!   'm' is a temporary value.
!   'i' is the index for the loops.
    real(c_double) m
    integer i
    
!   Here is the actual computation:
    c(1) = c(1) / b(1)
    x(1) = x(1) / b(1)
!   This is the syntax for a 'for' loop in Fortran.
!   Indexing for arrays in fortran starts at 1
!   instead of starting at 0 like it does in Python.
!   Arrays are accessed using parentheses
!   instead of brackets.
    do i = 1,n-2
        m = 1.0D0 / (b(i+1) - a(i) * c(i))
        c(i+1) = c(i+1) * m
        x(i+1) = x(i+1) - a(i) * x(i)
        x(i+1) = x(i+1) * m
!   Note that you have to explicitly end the loop.
    enddo
    x(n) = (x(n) - a(n-1) * x(n-1)) / (b(n) - a(n-1) * c(n-1))
    do i = n-1,1,-1
        x(i) = x(i) - c(i) * x(i+1)
    enddo
!   You must also explicitly end the function or subroutine.
end subroutine ftridiag

end module
